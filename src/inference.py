# src/inference.py

import os
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast
from utils.logger import logger

# -----------------------------
# 配置
# -----------------------------
MODEL_DIR = "output/model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 128
NUM_LABELS = 2  # SST-2 二分类

# -----------------------------
# 加载模型和 tokenizer
# -----------------------------
def load_model_tokenizer(model_dir=MODEL_DIR, device=DEVICE, num_labels=NUM_LABELS):
    logger.info(f"Loading model from {model_dir} to device {device}")
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    model.to(device)
    model.eval()
    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer

# -----------------------------
# 预测函数
# -----------------------------
def predict(texts, model, tokenizer, max_length=MAX_LENGTH, device=DEVICE):
    """
    texts: list of str
    returns: list of predicted labels
    """
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).cpu().tolist()

    for t, p in zip(texts, preds):
        print(f"[PREDICT] Text: {t} => Predicted label: {p}")
        logger.info(f"Text: {t} => Predicted label: {p}")

    return preds

# -----------------------------
# CLI 测试
# -----------------------------
if __name__ == "__main__":
    model, tokenizer = load_model_tokenizer()
    sample_texts = [
        "This movie was fantastic! I really enjoyed it.",
        "The film was boring and too long."
    ]
    predict(sample_texts, model, tokenizer)
