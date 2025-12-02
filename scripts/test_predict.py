# scripts/test_predict.py
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import boto3
import json
from src.utils.logger import logger

# ============================
# 配置
# ============================
endpoint_name = "nlp-sentiment-endpoint"  # 替换为你部署的 Endpoint 名称
region_name = "ca-central-1"              # 替换为你的 AWS 区域

# 初始化 SageMaker Runtime 客户端
runtime_client = boto3.client("sagemaker-runtime", region_name=region_name)

# ============================
# 示例输入
# ============================
sample_texts = [
    "This movie is fantastic! I really enjoyed it.",
    "The film was boring and too long."
]

# ============================
# 构造 Payload 并调用 Endpoint
# ============================
for text in sample_texts:
    payload = {"inputs": text}  # HuggingFace Inference API 标准格式
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload)
    )
    
    result = json.loads(response["Body"].read().decode())
    
    # 打印和记录日志
    print(f"[INPUT] {text}")
    print(f"[OUTPUT] {result}")
    logger.info(f"[INPUT] {text}")
    logger.info(f"[OUTPUT] {result}")
