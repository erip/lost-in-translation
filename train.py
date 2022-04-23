#!/usr/bin/env python3

from pathlib import Path
from argparse import ArgumentParser

import torch

from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding


def read_data(text_file, label_file, tokenizer):
    text = [line.strip() for line in open(text_file, encoding="utf-8")]
    labels = [int(line.strip()) for line in open(label_file, encoding="utf-8")]
    src_encoded = tokenizer(text, truncation=True)
    return ClassificationDataset(src_encoded, labels)


class ClassificationDataset(Dataset):
    def __init__(self, encoded_texts, labels):
        self.encoded_texts = encoded_texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encoded_texts.items()}
        item["labels"] = self.labels[idx]
        return item


def setup_argparse():
    parser = ArgumentParser()
    parser.add_argument("--training-text-file", type=Path, help="The newline-delimited file containing text (original or translated) to use to train")
    parser.add_argument("--training-label-file", type=Path, help="The newline-delimited file containing the labels corresponding to the labels in the training text file")
    parser.add_argument("--develop-text-file", type=Path, help="The newline-delimited file containing text (original or translated) to use to validate")
    parser.add_argument("--develop-label-file", type=Path, help="The newline-delimited file containing the labels corresponding to the labels in the develop text file")
    parser.add_argument("--output-dir", type=Path, required=True, help="The directory where checkpoints should be saved")
    parser.add_argument("-b", "--base-model", default="xlm-roberta-base", help="The base HuggingFace model to be finetuned.")
    parser.add_argument("-tbsz", "--train-batch-size", default=8, type=int, help="The training batch size")
    parser.add_argument("-ebsz", "--eval-batch-size", default=8, type=int, help="The evaluation batch size")
    return parser

if __name__ == "__main__":
    args = setup_argparse().parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    train_ds = read_data(args.training_text_file, args.training_label_file, tokenizer)
    eval_ds = read_data(args.develop_text_file, args.develop_label_file, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model,
        TrainingArguments(
          output_dir=args.output_dir
          do_train=True,
          do_eval=True,
          evaluation_strategy="steps",
          eval_steps=500,
          per_device_train_batch_size=args.train_batch_size,
          per_device_eval_batch_size=args.eval_batch_size,
          save_total_limit=2,
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )
    print(trainer.train())
