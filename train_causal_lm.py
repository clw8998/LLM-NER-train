import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
import jsonyx as json
import torch
from collections import defaultdict
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from datasets import Dataset, DatasetDict, Features, Value
import math
import re, os
import config
from seqeval.metrics import f1_score
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

config.set_seed()

with open('./data/test.pickle', 'rb') as file:
    test_data = pickle.load(file)
with open('./data/train.pickle', 'rb') as file:
    train_data = pickle.load(file)

grouped_test = defaultdict(dict)
for s in test_data:
    ctx   = s["context"]
    qname = config.ALIAS.get(s["question"], s["question"])
    grouped_test[ctx][qname] = s["answer"]

def bio_to_entities(text: str, tags: list[str]) -> list[str]:
    entities, start = [], None
    for i, tag in enumerate(tags + ['O']):
        if tag == 'B':
            start = i
        elif tag != 'I' and start is not None:
            entities.append(text[start:i])
            start = None
    return entities

def build_nested_dict(data: list[dict]) -> dict:
    questions = list(dict.fromkeys(d['question'] for d in data))
    result = defaultdict(lambda: {q: [] for q in questions})
    for d in data:
        ctx = d['context']
        q = config.ALIAS.get(d["question"], d["question"])
        result[ctx][q] = bio_to_entities(ctx, d['answer'])
    return dict(result)

model_name = "Qwen/Qwen3-1.7B"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name, 
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(device)

annotation_rules = config.annotation_rules
all_atts = config.all_atts

def process_nested_dict(nested_dict: dict) -> list[dict]:
    processed_data = []
    for context, answers in nested_dict.items():
        ordered_answer = {
            k: answers[k]
            for k in all_atts
        }
        messages = [
            {
                "role": "user",
                "content": config.PROMPT_TEMPLATE.format(ctx=context)
            },
            {   "role": "assistant", 
                "content": "```json\n" + json.dumps(ordered_answer, ensure_ascii=False, indent=1, indent_leaves=False) + "\n```"
            }
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False 
        )
        processed_data.append({"text": text})
    return processed_data

nested_train = build_nested_dict(train_data)

train_list = process_nested_dict(nested_train)

train_dataset = Dataset.from_list(train_list)

dataset = DatasetDict({
    "train": train_dataset,
})

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False, 
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    use_rslora=True,
)

exp_name = "qwen3_1.7b_causal_lm_ner"
epochs = 20
batch_size = 4
learning_rate = 5e-5
use_bf16 = True
gradient_accumulation_steps = 32 // batch_size
lr_scheduler_type = "cosine"
warmup_ratio = 0.05
weight_decay = 0.01
eval_steps = 50
temperature = 1e-5
model.generation_config.temperature = temperature

num_train_samples = len(train_list)

steps_per_epoch = math.ceil(num_train_samples / (batch_size * gradient_accumulation_steps))
total_steps = steps_per_epoch * epochs

training_args = SFTConfig(
    output_dir=f"./result/{exp_name}",
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    warmup_ratio=warmup_ratio,
    logging_steps=20,
    save_only_model=True,
    save_strategy="epoch",
    eval_strategy="epoch",
    weight_decay=weight_decay,
    report_to="none",
    push_to_hub=False,
    bf16=use_bf16,
    gradient_accumulation_steps=gradient_accumulation_steps,
    max_length=512,
)

def mask_tokens_inside_brackets(input_ids: torch.LongTensor, tokenizer):
    bs, _ = input_ids.size()
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for b in range(bs):
        text = tokenizer.decode(input_ids[b])
        m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.S)
        if not m:
            continue
        json_block = m.group(1)
        json_start = m.start(1)
        spans = [(s.start(1)+json_start, s.end(1)+json_start)
                 for s in re.finditer(r"\[(.*?)\]", json_block, flags=re.S)]
        if not spans:
            continue
        enc = tokenizer(text,
                        return_offsets_mapping=True,
                        add_special_tokens=False)
        for tok_idx, (tok_s, tok_e) in enumerate(enc["offset_mapping"]):
            if tok_s < 0:
                continue
            if any(tok_s < e and tok_e > s for s, e in spans):
                mask[b, tok_idx] = True

    assert mask.shape == input_ids.shape, "Mask shape must match input_ids shape"
    return mask

find_all   = config.find_all
make_O     = config.make_O
add_prefix = config.add_prefix
fill = config.fill

def compute_metrics(_):
    torch.cuda.empty_cache()
    def raw_to_dict(raw: str) -> dict:
        try:
            start = raw.index("{")
            end   = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except Exception:
            print(f"Error parsing raw text: {raw}")
            return {attr: [] for attr in all_atts}
    BATCH_SIZE   = 32
    MAX_NEW_TOKS = 256
    TEMPERATURE  = 1e-5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer.pad_token_id = model.generation_config.pad_token_id  # 151643
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"  
    pred_cache = {}
    uncached = []
    for ctx in grouped_test:
        user_msg = config.PROMPT_TEMPLATE.format(ctx=ctx)
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_msg}
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        uncached.append((ctx, prompt))

    if uncached:
        for start in tqdm(range(0, len(uncached), BATCH_SIZE), desc="Generating"):
            batch = uncached[start:start + BATCH_SIZE]
            ctx_batch, prompt_batch = zip(*batch)

            model_inputs = tokenizer(
                list(prompt_batch),
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            with torch.inference_mode():
                out_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=MAX_NEW_TOKS,
                    temperature=TEMPERATURE,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,   # now 151643
                    eos_token_id=tokenizer.eos_token_id,   # 151645
                )

            for i, ctx in enumerate(ctx_batch):
                input_len = model_inputs.input_ids[i].shape[0]
                raw = tokenizer.decode(out_ids[i, input_len:], skip_special_tokens=True)
                pred_cache[ctx] = raw_to_dict(raw)

    y_true, y_pred = [], []
    for ctx, attr_dict in grouped_test.items():
        for attr in all_atts:
            truth_tags = add_prefix(attr_dict[attr], attr) if attr in attr_dict else make_O(len(ctx))
            y_true.append(truth_tags)

        pred_attr_tags = {attr: make_O(len(ctx)) for attr in all_atts}
        pred_dict = pred_cache.get(ctx, {})
        if isinstance(pred_dict, dict):
            for attr, ents in pred_dict.items():
                if attr not in all_atts or not isinstance(ents, list):
                    continue
                for ent in ents:
                    if not isinstance(ent, str) or not ent.strip():
                        continue
                    for pos in find_all(ctx, ent):
                        fill(pred_attr_tags[attr], pos, len(ent), attr)

        for attr in all_atts:
            y_pred.append(pred_attr_tags[attr])

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"eval_macro_f1": macro_f1}

class MyTrainer(SFTTrainer):
    def evaluate(self, *args, **kwargs):
        metrics = compute_metrics(None)
        self.log(metrics)
        return metrics
    
# trainer = SFTTrainer(
trainer = MyTrainer(
    model=model,
    args=training_args,
    peft_config=peft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["train"],
    processing_class=tokenizer,
)

trainer.train()
