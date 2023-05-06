import torch

def multi_class_accuracy(preds: torch.Tensor, target: torch.Tensor) -> float:
    return ((torch.argmax(preds, dim=1) == torch.argmax(target, dim=1)) * 1).sum()\
        / target.shape[0]