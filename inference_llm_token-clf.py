import pickle
from collections import defaultdict
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from biqwen import Qwen3ForTokenClassification
from peft import PeftModel
from typing import List, Dict

device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = 'Qwen/Qwen3-1.7B'
lora_path = './result/Qwen3-1.7B-token-clf/checkpoint-110000'
label2id = {"O": 0, "I": 1, "B": 2}
sep_token = '[SEP]'

all_atts = [
    "品牌", "名稱", "產品","產品序號","顏色","材質","對象與族群",
    "適用物體、事件與場所","特殊主題","形狀","圖案","尺寸","重量","容量",
    "包裝組合","功能與規格",
]

with open("./data/test.pickle", "rb") as f:
    raw_data = pickle.load(f)
grouped = [f"{d['question']}{sep_token}{d['context']}" for d in raw_data]

max_length = 128
batch_size = 32

label2id = {"O": 0, "I": 1, "B": 2}
id2label_default = {v: k for k, v in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

base_model = Qwen3ForTokenClassification.from_pretrained(
    base_model,
    num_labels=len(label2id),
    id2label=id2label_default,
    label2id=label2id,
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, lora_path).to(device).eval()
id2label = getattr(model.config, "id2label", id2label_default)

@torch.inference_mode()
def predict_entities_after_sep(inputs_list: List[str], batch_size: int = batch_size) -> List[List[Dict]]:
    all_results: List[List[Dict]] = []
    n = len(inputs_list)

    with tqdm(total=n, desc="Infer", unit="sample") as pbar:
        for start in range(0, n, batch_size):
            batch_inputs = inputs_list[start:start + batch_size]

            enc = tokenizer(
                batch_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True
            )
            offsets = enc.pop("offset_mapping")
            enc = {k: v.to(device) for k, v in enc.items()}

            logits = model(**enc).logits
            pred_ids = logits.argmax(-1).cpu()
            attn = enc["attention_mask"].cpu()

            for i, text in enumerate(batch_inputs):
                spans, cur = [], None
                for j, m in enumerate(attn[i].tolist()):
                    if m == 0:
                        break
                    s, e = offsets[i][j].tolist()
                    if s == e:
                        continue
                    tag = id2label[int(pred_ids[i][j])]
                    if tag == "B":
                        if cur: spans.append(cur)
                        cur = {"start": s, "end": e}
                    elif tag == "I" and cur:
                        cur["end"] = e
                    else:
                        if cur: spans.append(cur)
                        cur = None
                if cur:
                    spans.append(cur)

                k = text.find(sep_token)
                sep_end = (k + len(sep_token)) if k != -1 else 0
                spans = [sp for sp in spans if sp["start"] >= sep_end]

                for sp in spans:
                    sp["text"] = text[sp["start"]:sp["end"]]
                    sp["start_rel"] = sp["start"] - sep_end
                    sp["end_rel"]   = sp["end"]   - sep_end

                all_results.append(spans)

            pbar.update(len(batch_inputs))

    return all_results

def infer_and_aggregate(texts: List[str]) -> Dict[str, Dict[str, List[str]]]:
    def empty_attr_dict():
        return {att: [] for att in all_atts}

    results: Dict[str, Dict[str, List[str]]] = defaultdict(empty_attr_dict)

    spans_list = predict_entities_after_sep(texts, batch_size=batch_size)

    for text, spans in zip(texts, spans_list):
        attr, ctx = text.split(sep_token, 1)
        entities = [s["text"] for s in spans]
        results[ctx][attr].extend(entities)

    return dict(results)

if __name__ == "__main__":
    results = infer_and_aggregate(grouped)
    for context, attrs in results.items():
        print(f"Context: {context}")
        for attr, entities in attrs.items():
            print(f"  {attr}: {entities}")
        print()
