import logging
import os
from datetime import datetime

# 日志目录和文件
logs_path = os.path.join(os.getcwd(), "output", "logs")
os.makedirs(logs_path, exist_ok=True)
log_file = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file_path = os.path.join(logs_path, log_file)

# logger 对象
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 文件处理器
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(
    logging.Formatter(
        "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
    )
)
logger.addHandler(file_handler)

# 控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter(
        "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
    )
)
logger.addHandler(console_handler)

# 测试
logger.info("Logger is set up and ready!")
