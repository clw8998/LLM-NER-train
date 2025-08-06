import pickle
from collections import defaultdict
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from biqwen import Qwen3ForTokenClassification
from peft import PeftModel


device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = 'Qwen/Qwen3-1.7B'
lora_path = './result/Qwen3-1.7B-token-clf/checkpoint-110000'
label2id = {"O": 0, "I": 1, "B": 2}
sep_token = '[SEP]'
all_atts = annotation_keys = [
    "品牌", "名稱", "產品","產品序號","顏色","材質","對象與族群",
    "適用物體、事件與場所","特殊主題","形狀","圖案","尺寸","重量","容量",
    "包裝組合","功能與規格",
]

with open("./data/test.pickle", "rb") as f:
    raw_data = pickle.load(f)
grouped = [f"{d['question']}{sep_token}{d['context']}" for d in raw_data]

tokenizer = AutoTokenizer.from_pretrained(base_model)


base = Qwen3ForTokenClassification.from_pretrained(
    base_model,
    num_labels=len(label2id), 
    id2label={v: k for k, v in label2id.items()}, 
    label2id=label2id,
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base, lora_path).to(device).eval()

for param in model.parameters():
    if not param.is_contiguous():
        param.data = param.contiguous()


def run_inference(texts, batch_size=32, max_len=128):
    results = defaultdict(lambda: {att: [] for att in all_atts})
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Inferencing"):
        batch = texts[i:i+batch_size]
        model_input = tokenizer(batch, return_offsets_mapping=True, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        offsets = model_input.pop("offset_mapping").cpu().tolist()
        
        with torch.no_grad():
            preds = model(**model_input).logits.argmax(-1).cpu().tolist()
        
        for sent, ids, off_map in zip(batch, preds, offsets):
            attr = sent.split(sep_token, 1)[0].strip()
            context = sent.split(sep_token, 1)[1].strip()
            
            cur_span = None
            for (start, end), tag in zip(off_map, map(model.config.id2label.get, ids)):
                if start == end: continue
                if tag == "B":
                    if cur_span and (ent := sent[cur_span["start"]:cur_span["end"]].strip()) and ent not in results[context][attr]:
                        results[context][attr].append(ent)
                    cur_span = {"start": start, "end": end}
                elif tag == "I" and cur_span:
                    cur_span["end"] = end
                elif cur_span and (ent := sent[cur_span["start"]:cur_span["end"]].strip()) and ent not in results[context][attr]:
                    results[context][attr].append(ent)
                    cur_span = None
            if cur_span and (ent := sent[cur_span["start"]:cur_span["end"]].strip()) and ent not in results[context][attr]:
                results[context][attr].append(ent)
    
    return results

if __name__ == "__main__":
    results = run_inference(grouped)
    # beautify the output
    for context, attrs in results.items():
        print(f"Context: {context}")
        for attr, values in attrs.items():
            print(f"  {attr}: {', '.join(values)}")
        print()
    # print(results)
