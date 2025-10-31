# Metrics and label conversion functions 
import numpy as np
import config

def to_01_labels(y_pm1):
    return (y_pm1 > 0).astype(np.uint8)


def to_pm1_labels(y01):
    return np.where(y01 == 1, 1, -1).astype(np.int32)


def accuracy_score(y_true01, y_pred01):
    y_true01 = np.asarray(y_true01)
    y_pred01 = np.asarray(y_pred01)
    return float(np.mean(y_true01 == y_pred01))


def precision_recall_f1(y_true01, y_pred01):
    eps=config.EPS #(EPS = 1e-12 # safeguard Avoid division by 0)
    y_true01 = np.asarray(y_true01)
    y_pred01 = np.asarray(y_pred01)
    tp = np.sum((y_true01 == 1) & (y_pred01 == 1))
    fp = np.sum((y_true01 == 0) & (y_pred01 == 1))
    fn = np.sum((y_true01 == 1) & (y_pred01 == 0))
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    return float(precision), float(recall), float(f1)