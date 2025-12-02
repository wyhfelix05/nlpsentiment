# scripts/sagemaker_train.py
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import sagemaker
from sagemaker.huggingface import HuggingFace
import os
from src.utils.logger import logger  # 使用你已有的 logger

# ============================
# 配置
# ============================
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::608841106628:role/nlp-sst2-sagemaker-role"  # 如果在本地运行，需要替换为 IAM Role ARN
bucket = "nlp-project-1201"
prefix = "nlp-sentiment"

# S3 数据路径
train_input = "s3://nlp-project-1201/nlp-sentiment/data/processed/"

# ============================
# 训练开始日志
# ============================
logger.info("Initializing SageMaker HuggingFace Estimator...")

# ============================
# HuggingFace Estimator 配置
# ============================
huggingface_estimator = HuggingFace(
    entry_point="train.py",          # 训练入口脚本
    source_dir="src",                # 源码目录
    instance_type="ml.g4dn.xlarge",  # GPU 训练实例
    instance_count=1,                # 实例数量
    role=role,
    transformers_version="4.28.1",     # Transformers 版本
    pytorch_version="2.0.0",           # PyTorch 版本
    py_version="py310",              # Python 版本
    hyperparameters={                # 超参数
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "max_length": 128
    },
    sagemaker_session=sagemaker_session
)

logger.info("Estimator configured successfully.")

# ============================
# 启动训练任务并接入日志
# ============================
logger.info(f"Starting training job with data from {train_input} ...")

# wait=True 阻塞脚本直到训练完成
# logs=True 会实时打印 CloudWatch 日志到终端
huggingface_estimator.fit({"train": train_input}, wait=True, logs=True)

logger.info("Training job finished. Check CloudWatch for detailed logs.")
logger.info(f"Trained model artifacts saved at: {huggingface_estimator.model_data}")
