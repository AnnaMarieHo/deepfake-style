import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def compute_metrics(y_true, y_prob):
    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    return acc, auc
