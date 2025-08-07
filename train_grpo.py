from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
import torch, re, os, gc, math, pickle
from collections import defaultdict
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from datasets import Dataset, DatasetDict
from seqeval.metrics import f1_score
import config
from collections import defaultdict
from tqdm import tqdm
import jsonyx as json

config.set_seed()

os.environ["WANDB_DISABLED"] = "true" # Set to "false" to enable Weights & Biases logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"

with open('./data/train.pickle', 'rb') as file:
    train_data = pickle.load(file)

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

base_model_name = "Qwen/Qwen3-1.7B"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

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
            # {"role": "system", "content": "You are a specialist in e-commerce NER"},
            {
                "role": "user",
                "content": config.PROMPT_TEMPLATE.format(ctx=context)
            },
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        processed_data.append({"prompt": text, 'gt': ordered_answer, 'context': context})
    return processed_data

nested_train = build_nested_dict(train_data)
train_list = process_nested_dict(nested_train)

# train_list = train_list[:1000]

train_dataset = Dataset.from_list(train_list)

print(f"Train dataset size: {len(train_dataset)}")

dataset = DatasetDict({
    "train": train_dataset,
})

with open("./data/test.pickle", "rb") as f:
    raw_test = pickle.load(f)

grouped_test = defaultdict(dict)
for s in raw_test:
    ctx   = s["context"]
    qname = config.ALIAS.get(s["question"], s["question"])
    grouped_test[ctx][qname] = s["answer"]

find_all   = config.find_all
make_O     = config.make_O
add_prefix = config.add_prefix
fill = config.fill

def f1_reward_func(completions: list[str], gt: list[dict], context: list[str], **kwargs) -> list[float]:
    rewards = []
    for pred_text, true_attr_tags, ctx in zip(completions, gt, context):
        try:
            start = pred_text.index("{")
            end   = pred_text.rindex("}") + 1
            pred_dict = json.loads(pred_text[start:end])
        except Exception:
            rewards.append(-0.5)
            continue

        try: 
            y_true, y_pred = [], []

            for attr in all_atts:
                true_tags = make_O(len(ctx))
                for ent in true_attr_tags.get(attr, []):
                    for pos in find_all(ctx, ent):
                        fill(true_tags, pos, len(ent), attr)
                y_true.append(true_tags)

                pred_tags = make_O(len(ctx))
                if isinstance(pred_dict, dict):
                    ents = pred_dict.get(attr, [])
                    if isinstance(ents, list):
                        for ent in ents:    
                            if not ent:
                                continue
                            for pos in find_all(ctx, ent):
                                fill(pred_tags, pos, len(ent), attr)
                y_pred.append(pred_tags)
        
            try:
                if y_true == y_pred:
                    f1 = 1.0
                else:
                    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            except Exception:
                f1 = 0.0
        except Exception as e:
            f1 = -0.5
        rewards.append(float(f1))

    return rewards

MAX_NEW_TOKS = 256

def thinking_reward_func(completions: list[str], tokenizer, X_max: int = 1024, **kwargs) -> list[float]:
    X_max = X_max - 256
    rewards = []
    for pred_text in completions:
        content = pred_text
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if match:
            think_content = match.group(1).strip()
            token_count = len(tokenizer.encode(think_content, add_special_tokens=False))
            reward = math.log(1 + token_count) / math.log(1 + X_max)
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards

class MemoryCleanupCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            torch.cuda.empty_cache()

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

class MyGRPOTrainer(GRPOTrainer):
    def evaluate(self, *args, **kwargs):
        metrics = compute_metrics(None)
        self.log(metrics)
        return metrics

# test diffrent beta and epsilon values
# i = 0
# for beta in [0.0, 0.2, 0.8]:
for beta in [0.0]:
    # for epsilon in [0.0, 0.2, 0.8]:
    for epsilon in [0.8]:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        model = base_model
        model.eval()

        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False, 
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            use_rslora=True,
        )

        i = 0
        while os.path.exists(f"./result/qwen3_1.7b_grpo_ner_v{i}"):
            i += 1
            
        exp_name = f"qwen3_1.7b_grpo_ner_v{i}"
        
        epochs = 10
        batch_size = 4
        num_generations = 4
        learning_rate = 1e-5
        use_bf16 = True
        gradient_accumulation_steps = 32 // batch_size
        lr_scheduler_type = "linear"
        warmup_ratio = 0.05
        weight_decay = 0.01
        beta = beta
        max_completion_length=MAX_NEW_TOKS
        temperature=0.6
        epsilon=epsilon

        training_args = GRPOConfig(
            output_dir=f"./result/{exp_name}",
            num_generations=num_generations,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            logging_steps=20,
            save_strategy="epoch",
            # save_steps=100,
            eval_strategy="epoch",
            weight_decay=weight_decay,
            report_to="none",  # Set to "wandb" to enable Weights & Biases logging
            push_to_hub=False,
            bf16=use_bf16,
            gradient_accumulation_steps=gradient_accumulation_steps,
            beta=beta,
            max_completion_length=max_completion_length,
            scale_rewards=False,
            temperature=temperature,
            generation_kwargs={"temperature": temperature},
            epsilon=epsilon,
            # optim="adamw_8bit",
            save_only_model=True,
        )

        model.generation_config.temperature = temperature

        # Initialize trainer
        # trainer = GRPOTrainer(
        trainer = MyGRPOTrainer(
            model=model,
            reward_funcs=[
                            f1_reward_func, 
                            # lambda completions, **kw: thinking_reward_func(completions, tokenizer=tokenizer, X_max=MAX_NEW_TOKS, **kw),
                        ],
            args=training_args,
            peft_config=peft_config if peft_config else None,
            train_dataset=dataset["train"],
            eval_dataset=dataset["train"],
            processing_class=tokenizer,
            callbacks=[MemoryCleanupCallback()],
        )

        print("GenerationConfig:", trainer.model.generation_config)
        print("Temperature:", trainer.model.generation_config.temperature)

        # Start training
        trainer.train()
        
        # clean all the memory
        del trainer
        del model
        del base_model
        del tokenizer
        del peft_config
        gc.collect()
        torch.cuda.empty_cache()
