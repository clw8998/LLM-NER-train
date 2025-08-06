# -*- coding: utf-8 -*-

import random
import numpy as np
import evaluate
import torch
import json
from transformers import set_seed
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from biqwen import Qwen3ForTokenClassification
from transformers import DataCollatorForTokenClassification
import os
from seqeval.metrics import classification_report
from biqwen import Qwen3ForTokenClassification
from peft import PeftModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

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

base_model = 'Qwen/Qwen3-1.7B'
lora_path = './result/Qwen3-1.7B-token-clf/checkpoint-110000'

test_data_name = 'test_token-clf'

def load_ner_dataset():
    ret = {}
    for split_name in [test_data_name]:
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

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

base = Qwen3ForTokenClassification.from_pretrained(
    base_model,
    num_labels=len(label_list), 
    id2label=id2label, 
    label2id=label2id, 
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base, lora_path)

for param in model.parameters():
    if not param.is_contiguous():
        param.data = param.contiguous()

eval_args = TrainingArguments(
    output_dir="./result/eval_only",
    per_device_eval_batch_size=32,
    report_to="none",
    do_train=False,
    do_eval=True
)

tester = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=tokenized_ds[test_data_name],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    
)

results = tester.evaluate()
print(json.dumps(results, indent=2, ensure_ascii=False))

# remove output directory after evaluation
import shutil
shutil.rmtree("./result/eval_only", ignore_errors=True)