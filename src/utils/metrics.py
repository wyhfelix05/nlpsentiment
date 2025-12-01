from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_classification_metrics(y_true, y_pred, average='binary'):
    """
    计算分类指标并打印
    
    Args:
        y_true (list or np.array): 真实标签
        y_pred (list or np.array): 模型预测标签
        average (str): 'binary', 'macro', 'micro', 'weighted'
    
    Returns:
        dict: 包含 accuracy, precision, recall, f1
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)

    # 打印到终端
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

    return metrics

def compute_confusion(y_true, y_pred):
    """
    计算混淆矩阵并打印
    
    Args:
        y_true (list or np.array): 真实标签
        y_pred (list or np.array): 模型预测标签
        
    Returns:
        np.array: 混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    return cm

