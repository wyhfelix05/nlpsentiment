# NLP Sentiment Classification with SageMaker Deployment

## Table of Contents

1. [Description](#description)
2. [Tech Stack](#tech-stack)
3. [Project Structure (Simplified / Key Files)](#project-structure-simplified--key-files)
4. [Installation / Setup](#installation--setup)
5. [Deployment](#deployment)
6. [Configuration](#configuration)
7. [License](#license)


## Description

This project uses a custom text dataset to build a natural language processing (NLP) model for sentiment analysis. This project demonstrates a streamlined, production-ready NLP workflow, spanning model training, evaluation, and deployment on SageMaker.

The focus is not only on training and evaluating the NLP model, but also on deploying and serving it in the cloud. The trained model is packaged and deployed on AWS SageMaker, where it can handle real-time inference requests. Users can send JSON requests with text input and receive sentiment predictions instantly through the SageMaker endpoint.

This project covers the essential steps of an end-to-end NLP workflow:

- Data preprocessing and tokenization for NLP inputs
- Model training, validation, and evaluation
- Packaging the trained model and tokenizer for deployment
- Deploying the model to an AWS SageMaker endpoint
- Configuring the endpoint for real-time inference
- Designing clear JSON request/response structures for predictions


The result is a practical, cloud-based NLP prototype that demonstrates how text-based machine learning models can be trained, deployed, and served using AWS infrastructure.

## Tech Stack

### Languages & Libraries
- Python
- NumPy
- Pandas
- Scikit-learn
- Transformers (Hugging Face)
- Datasets (Hugging Face)
- PyTorch
- Matplotlib
- Seaborn

### Cloud Services
- AWS SageMaker
- AWS S3
- AWS IAM

### Development Tools
- VSCode
- Git / GitHub
- AWS CLI

## Project Structure (Simplified / Key Files)

```plaintext
project/
├── config/
│   └── config.yaml           # 配置文件
├── data/                     # 数据存放目录
├── output/
│   ├── logs/                 # 训练日志
│   └── model/                # 保存的模型和 tokenizer
├── scripts/
│   ├── deploy_endpoint.py    # 部署 SageMaker 端点
│   ├── delete_endpoint.py    # 删除 SageMaker 端点
│   └── sagemaker_train.py    # 训练脚本
├── src/
│   ├── train.py              # 训练逻辑
│   ├── inference.py          # 推理逻辑
│   └── utils/                # 工具模块（日志、评估等）
├── README.md
└── requirements.txt
```

## Installation / Setup

```bash
# 1. Clone the repository
git clone https://github.com/wyhfelix05/nlpsentiment
cd project

# 2. Create a virtual environment
python -m venv nlpenv

# Activate the environment
# On Windows
nlpenv\Scripts\activate
# On macOS/Linux
source nlpenv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Deployment

This project deploys the trained NLP model to **AWS SageMaker** for real-time inference. The deployment workflow is as follows:

1. **Train the model**  
   You can run `scripts/sagemaker_train.py` to start a SageMaker training job using the prepared dataset and configuration in `config/config.yaml`.

2. **Package the model**  
   After training, the model and tokenizer are saved in `output/model/` and can be optionally containerized with Docker for SageMaker deployment.

3. **Deploy to SageMaker endpoint**  
   Run `scripts/deploy_endpoint.py` to create a SageMaker endpoint for real-time predictions.

4. **Test the endpoint**  
   Use `scripts/test_predict.py` to send JSON requests with text input to the SageMaker endpoint and receive sentiment predictions.

5. **Delete the endpoint**  
   When finished, run `scripts/delete_endpoint.py` to remove the SageMaker endpoint and avoid unnecessary costs.

## Configuration

The project uses a YAML configuration file to manage paths, model parameters, and preprocessing settings.

### Key configuration file

- `config/config.yaml`  
  Defines paths to input datasets, output artifacts (e.g., preprocessed data, trained model), and other settings such as tokenizer options and training parameters.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this project for personal or commercial purposes, provided that the original copyright notice and license are included.