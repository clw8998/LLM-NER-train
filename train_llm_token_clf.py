# -*- coding: utf-8 -*-

import random
import numpy as np
import evaluate
import torch
import json
from transformers import set_seed
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, TrainingArguments, Trainer
from biqwen import Qwen3ForTokenClassification
from transformers import DataCollatorForTokenClassification
import os
from seqeval.metrics import classification_report
from peft import get_peft_model, LoraConfig, TaskType  

os.environ["WANDB_DISABLED"] = "true" # Set to "false" to enable Weights & Biases logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

set_random_seed(2024)

train_data_name = 'train_token-clf'
test_data_name = 'test_token-clf'

exp_name = "Qwen3-1.7B-token-clf"
model_id = "Qwen/Qwen3-1.7B"
epochs = 20
learning_rate = 5e-5
batch_size = 32
gradient_accumulation_steps = 32 // batch_size
use_bf16 = True
torch_dtype = torch.bfloat16 if use_bf16 else torch.float32
lr_scheduler_type = 'cosine'
warmup_ratio = 0.05

def load_ner_dataset():
    ret = {}
    for split_name in [train_data_name, test_data_name]:
        data = []
        with open(f'./data/{split_name}.jsonl', 'r', encoding="utf-8") as reader:
            for line in reader:
                data.append(json.loads(line))
        ret[split_name] = Dataset.from_list(data)
    return DatasetDict(ret)

max_length = 128 
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["inputs"],
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )

    batch_labels = []
    for offsets, char_labels in zip(tokenized_inputs["offset_mapping"], examples["ner_tags"]):
        label_ids = []
        for start, end in offsets:
            if start == end:
                label_ids.append(-100)
            else:
                label_ids.append(char_labels[start])
        batch_labels.append(label_ids)

    tokenized_inputs["labels"] = batch_labels
    tokenized_inputs.pop("offset_mapping")
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    report = classification_report(
        y_true=true_labels,
        y_pred=true_predictions,
        output_dict=True
    )
    macro_precision = report["macro avg"]["precision"]
    macro_recall    = report["macro avg"]["recall"]
    macro_f1        = report["macro avg"]["f1-score"]

    return {
        "eval_precision": macro_precision,
        "eval_recall": macro_recall,
        "eval_f1": macro_f1,
    }

ds = load_ner_dataset()
label2id = {'O': 0, 'I': 1, 'B': 2}
id2label = {v: k for k, v in label2id.items()}
label_list = list(label2id.keys())

seqeval = evaluate.load("seqeval")

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

model = Qwen3ForTokenClassification.from_pretrained(
    model_id, num_labels=len(label_list), id2label=id2label, label2id=label2id, torch_dtype=torch_dtype
)

for param in model.parameters():
    if not param.is_contiguous():
        param.data = param.contiguous()

peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS, 
    inference_mode=False, 
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    r=16,
    lora_alpha=32, 
    lora_dropout=0.05,
    use_rslora=True,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=f"./result/{exp_name}",
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=5e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    push_to_hub=False,
    report_to="none",  # Set to "wandb" to enable Weights & Biases logging
    gradient_accumulation_steps=gradient_accumulation_steps,
    bf16=use_bf16,
    logging_steps=100,
    warmup_ratio=warmup_ratio,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds[train_data_name],
    eval_dataset=tokenized_ds[test_data_name],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
