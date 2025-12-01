import os
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from transformers import BertTokenizerFast
from datasets import load_from_disk
from utils.metrics import compute_classification_metrics

# 假设你的 logger 已经在 logger.py 或 train.py 顶部定义
from utils.logger import logger


def main(debug=False, debug_size=5000):
    # -----------------------------
    # 配置
    # -----------------------------
    model_name = "bert-base-uncased"
    output_dir = "output/model"
    num_labels = 2
    batch_size = 16
    epochs = 3
    max_length = 128

    # -----------------------------
    # 加载 tokenizer
    # -----------------------------
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    # -----------------------------
    # 加载数据
    # -----------------------------
    dataset_dir = "data/processed"
    datasets = load_from_disk(dataset_dir)

    if debug:
        print(f"[DEBUG] Using first {debug_size} examples for local testing.")
        logger.info(f"[DEBUG] Using first {debug_size} examples for local testing.")
        datasets["train"] = datasets["train"].select(range(debug_size))
        datasets["validation"] = datasets["validation"].select(
            range(min(debug_size, len(datasets["validation"])))
        )

    # -----------------------------
    # 初始化模型
    # -----------------------------
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # -----------------------------
    # 定义训练参数
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # -----------------------------
    # 定义 Trainer
    # -----------------------------
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        import numpy as np
        preds = np.argmax(logits, axis=-1)
        metrics = compute_classification_metrics(labels, preds, average="binary")
        logger.info(f"Validation metrics: {metrics}")
        return metrics

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # -----------------------------
    # 开始训练
    # -----------------------------
    logger.info("Training started.")
    print("Training started.")
    trainer.train()
    logger.info("Training finished.")
    print("Training finished.")

    # -----------------------------
    # 保存模型
    # -----------------------------
    trainer.save_model(output_dir)
    logger.info(f"Model saved to {output_dir}")
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main(debug=True, debug_size=1000)
