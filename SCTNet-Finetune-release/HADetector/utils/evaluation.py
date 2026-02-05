import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve
import numpy as np
from torch.utils.data import DataLoader
import math


def genertate_region_mask(masks ,batch_shape):
    """generate B 1 H W meaningful-region-mask for a batch of masks

    Args:
        batch_shape (_type_): _description_
    """
    meaningful_mask = torch.zeros_like(masks)
    print("batch_shape: ", batch_shape)
    meaningful_mask[ :, :batch_shape[0], :batch_shape[1]] = 1
        
    return meaningful_mask


def cal_confusion_matrix(predict, target, region_mask, threshold=0.5):
    """compute local confusion matrix for a batch of predict and target masks
    Args:
        predict (_type_): _description_
        target (_type_): _description_
        region (_type_): _description_
        
    Returns:
        TP, TN, FP, FN
    """
    predict = (predict > threshold).float()
    TP = torch.sum(predict * target * region_mask, dim=(1, 2, 3))
    TN = torch.sum((1-predict) * (1-target) * region_mask, dim=(1, 2, 3))
    FP = torch.sum(predict * (1-target) * region_mask, dim=(1, 2, 3))
    FN = torch.sum((1-predict) * target * region_mask, dim=(1, 2, 3))
    return TP, TN, FP, FN


def cal_F1(TP, TN, FP, FN):
    """compute F1 score for a batch of TP, TN, FP, FN
    Args:
        TP (_type_): _description_
        TN (_type_): _description_
        FP (_type_): _description_
        FN (_type_): _description_
    """
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    F1 = 2 * precision * recall / (precision + recall + 1e-8)
    # F1 = torch.mean(F1) # fuse the Batch dimension
    return F1


def cal_accuracy(TP, TN, FP, FN):
    """Compute accuracy for a batch of TP, TN, FP, FN
    Args:
        TP (torch.Tensor): True Positives
        TN (torch.Tensor): True Negatives
        FP (torch.Tensor): False Positives
        FN (torch.Tensor): False Negatives
    Returns:
        torch.Tensor: Accuracy
    """
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    return accuracy


def cal_iou(TP, TN, FP, FN):
    """Compute Intersection over Union (IoU) for a batch of TP, TN, FP, FN
    Args:
        TP (torch.Tensor): True Positives
        TN (torch.Tensor): True Negatives
        FP (torch.Tensor): False Positives
        FN (torch.Tensor): False Negatives
    Returns:
        torch.Tensor: IoU
    """
    iou = TP / (TP + FP + FN + 1e-8)
    return iou


def artifact_score(predict):
    """
    Compute the artifact score for a given prediction tensor.

    Args:
        predict (torch.Tensor): The input prediction tensor.

    Returns:
        float: The artifact score, which is the mean value of the normalized prediction tensor.
    """
    # Ensure the tensor is on the CPU
    predict = predict.cpu()
    
    # Normalize the tensor to the range [0, 1]
    predict_min = predict.min()
    predict_max = predict.max()
    normalized_predict = (predict - predict_min) / (predict_max - predict_min + 1e-8)
    
    # Compute the mean value of the normalized tensor
    artifact_score = normalized_predict.mean().item()
    
    return artifact_score


def artifact_score_nonlinear(predict, alpha=10.0):
    """
    Compute the non-linear artifact score (AS) based on Equation 11.
    
    AS = min(log(1 + alpha * AR), 1)
    
    Args:
        predict (torch.Tensor): The input prediction tensor (probability map or error map).
        alpha (float): Scaling factor, default is 10 as per the paper.

    Returns:
        float: The final artifact score [0, 1].
    """
    predict = predict.cpu()
    
    predict_min = predict.min()
    predict_max = predict.max()
    normalized_predict = (predict - predict_min) / (predict_max - predict_min + 1e-8)
    
    ar = normalized_predict.mean().item()
    
    log_value = math.log10(1 + alpha * ar)
    
    artifact_score = min(log_value, 1.0)
    
    return artifact_score


def cal_PAR(predict, threshold=0.5):
    predict = predict.cpu().numpy()
    binary_predict = (predict >= threshold).astype(int)
    pixel_artifact_ratio = np.mean(binary_predict)

    return pixel_artifact_ratio