"""
Evaluation metrics for perception and RL modules.
"""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def segmentation_metrics(y_true, y_pred):
    acc = accuracy_score(y_true.flatten(), y_pred.flatten())
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    return {'accuracy': acc, 'confusion_matrix': cm}

def rl_metrics(rewards):
    return {'mean_reward': np.mean(rewards), 'std_reward': np.std(rewards)}
