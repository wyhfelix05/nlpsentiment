import torch
import os
from datasets import load_from_disk

# -------------------------
# 模型保存
# -------------------------
def save_model(model, path):
    """
    保存模型权重
    Args:
        model: PyTorch 模型
        path: 保存路径
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def save_pretrained_model(model, tokenizer, path):
    """
    保存 Transformers 模型和 tokenizer
    Args:
        model: Transformers 模型
        tokenizer: 对应 tokenizer
        path: 保存路径
    """
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Pretrained model and tokenizer saved to {path}")

# -------------------------
# 模型加载
# -------------------------
def load_processed_dataset(path="data/processed", debug=False, debug_size=5000):
    """
    从磁盘加载已经处理好的数据集，并可选择调试模式下使用小数据。
    
    Args:
        path (str): 数据集存放路径
        debug (bool): 是否使用小数据
        debug_size (int): 小数据样本数
    Returns:
        datasets.DatasetDict
    """
    datasets = load_from_disk(path)
    
    if debug:
        datasets["train"] = datasets["train"][:debug_size]
        datasets["validation"] = datasets["validation"][:debug_size]
    
    return datasets

def load_model(model_class, path, device, **kwargs):
    """
    加载模型权重
    Args:
        model_class: 模型类
        path: 权重文件路径
        device: 'cpu' 或 'cuda'
        kwargs: 初始化参数
    Returns:
        model 实例
    """
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded from {path}")
    return model

# -------------------------
# 辅助函数
# -------------------------
def count_parameters(model):
    """
    统计模型可训练参数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def move_to_device(batch, device):
    """
    将 batch 数据移动到指定设备
    Args:
        batch: dict, batch 数据
        device: 'cpu' 或 'cuda'
    Returns:
        batch 移动到设备后的副本
    """
    return {k: v.to(device) for k, v in batch.items()}
