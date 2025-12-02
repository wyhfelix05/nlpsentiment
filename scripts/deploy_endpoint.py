# scripts/deploy_endpoint.py
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from src.utils.logger import logger

# ============================
# 配置
# ============================
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::608841106628:role/nlp-sst2-sagemaker-role" # 如果在本地，需要替换为 IAM Role ARN
bucket = "nlp-project-1201"

# 训练完成后的模型 S3 路径
# 可以在 huggingface_estimator.model_data 里找到，或者手动指定
model_data = "s3://sagemaker-ca-central-1-608841106628/huggingface-pytorch-training-2025-12-02-07-33-08-371/output/model.tar.gz"

# 部署配置
instance_type = "ml.m5.large"
initial_instance_count = 1
endpoint_name = "nlp-sentiment-endpoint"  # 自定义 Endpoint 名称

# ============================
# 日志记录
# ============================
logger.info("Initializing HuggingFaceModel for deployment...")

# ============================
# 创建 HuggingFaceModel 对象
# ============================
huggingface_model = HuggingFaceModel(
    model_data=model_data,
    role=role,
    transformers_version="4.28.1",
    pytorch_version="2.0.0",
    py_version="py310",
    sagemaker_session=sagemaker_session,
    env={"HF_TASK": "text-classification"}   
)


logger.info("Model object created successfully.")

# ============================
# 部署到 SageMaker Endpoint
# ============================
logger.info(f"Deploying model to endpoint: {endpoint_name} ...")

predictor = huggingface_model.deploy(
    initial_instance_count=initial_instance_count,
    instance_type=instance_type,
    endpoint_name=endpoint_name
)

logger.info(f"Endpoint {endpoint_name} deployed successfully.")
logger.info(f"Endpoint name: {predictor.endpoint_name}")

print(f"Deployment finished. Endpoint name: {predictor.endpoint_name}")
