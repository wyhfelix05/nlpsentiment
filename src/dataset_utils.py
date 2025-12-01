# src/dataset_utils.py

import os
from datasets import load_dataset, DatasetDict
from transformers import BertTokenizer
from typing import Optional, Tuple

# =========================================================
# 阶段 1：下载与基本检查
# =========================================================

def load_raw_dataset(dataset_name: str = "stanfordnlp/sst2") -> DatasetDict:
    """Download a dataset from Hugging Face hub."""
    print(f"[INFO] Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    print(dataset)
    return dataset


def clean_dataset(dataset: DatasetDict) -> DatasetDict:
    """Basic cleaning: strip whitespace, drop empty samples."""
    def clean_fn(example):
        text = example["sentence"].strip() if "sentence" in example else example["text"].strip()
        return {"text": text}

    for split in dataset.keys():
        dataset[split] = dataset[split].rename_column(
            "sentence" if "sentence" in dataset[split].column_names else "text",
            "text"
        )
        dataset[split] = dataset[split].map(clean_fn)
        dataset[split] = dataset[split].filter(lambda x: len(x["text"]) > 0)

    print("[INFO] Dataset cleaned.")
    return dataset


# =========================================================
# 阶段 2：Tokenization
# =========================================================

def load_tokenizer(tokenizer_name: str = "bert-base-uncased") -> BertTokenizer:
    print(f"[INFO] Loading tokenizer: {tokenizer_name}")
    return BertTokenizer.from_pretrained(tokenizer_name)


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer,
    max_length: int = 128
) -> DatasetDict:

    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    print("[INFO] Tokenizing dataset...")
    tokenized = dataset.map(tokenize_fn, batched=True)
    print("[INFO] Tokenization done.")

    return tokenized


# =========================================================
# 阶段 3：保存
# =========================================================

def save_dataset(dataset: DatasetDict, save_dir: str = "data/processed"):
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Saving tokenized dataset to: {save_dir}")
    dataset.save_to_disk(save_dir)
    print("[INFO] Dataset saved.")


# =========================================================
# 阶段 4：自动流水线（核心函数）
# =========================================================

def prepare_dataset(
    dataset_name: str = "stanfordnlp/sst2",
    tokenizer_name: str = "bert-base-uncased",
    save_dir: str = "data/processed",
    max_length: int = 128,
):
    """
    FULL PIPELINE:
        1. Download dataset
        2. Clean dataset
        3. Tokenize dataset
        4. Save to disk
    返回可直接用于训练的 HF DatasetDict
    """
    print("========== DATASET PREPARATION START ==========")

    # ---- 调用前面每个阶段 ----
    raw_dataset = load_raw_dataset(dataset_name)
    cleaned_dataset = clean_dataset(raw_dataset)
    tokenizer = load_tokenizer(tokenizer_name)
    tokenized_dataset = tokenize_dataset(cleaned_dataset, tokenizer, max_length)
    save_dataset(tokenized_dataset, save_dir)

    print("========== DATASET PREPARATION DONE ==========")
    return tokenized_dataset


# =========================================================
# 可选 CLI 用法
# =========================================================
if __name__ == "__main__":
    prepare_dataset()
