# scripts/delete_endpoint.py
import sys
import os
import boto3  # 用于删除 EndpointConfig

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import sagemaker
from src.utils.logger import logger

# ============================
# 配置
# ============================
sagemaker_session = sagemaker.Session()
sm_client = boto3.client("sagemaker")  # boto3 客户端

# 要删除的 Endpoint 名称（也是 EndpointConfig 名称）
endpoint_name = "nlp-sentiment-endpoint"

# ============================
# 日志记录
# ============================
logger.info(f"Deleting SageMaker endpoint: {endpoint_name} ...")

# ============================
# 删除 Endpoint
# ============================
try:
    sagemaker_session.delete_endpoint(endpoint_name)
    logger.info(f"Endpoint {endpoint_name} deleted successfully.")
    print(f"Endpoint {endpoint_name} deleted successfully.")
except Exception as e:
    logger.error(f"Failed to delete endpoint {endpoint_name}: {e}")
    print(f"Failed to delete endpoint {endpoint_name}: {e}")

# ============================
# 删除 Endpoint Configuration
# ============================
logger.info(f"Deleting SageMaker endpoint configuration: {endpoint_name} ...")
try:
    sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
    logger.info(f"Endpoint configuration {endpoint_name} deleted successfully.")
    print(f"Endpoint configuration {endpoint_name} deleted successfully.")
except sm_client.exceptions.ClientError as e:
    logger.error(f"Failed to delete endpoint configuration {endpoint_name}: {e}")
    print(f"Failed to delete endpoint configuration {endpoint_name}: {e}")
