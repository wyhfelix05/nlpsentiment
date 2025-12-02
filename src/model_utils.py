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


# -------------------------
# 模型加载
# -------------------------


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

