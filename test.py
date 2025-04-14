from collections import Counter
import numpy as np
y_true = [0, 1, 0, 0, 1]   # 真實標籤
y_pred = [0, 0, 1, 1, 1]   # 預測標籤
def f1_score(y_true, y_pred):
    # Get unique classes
    classes = np.unique(np.concatenate((y_true, y_pred)))
    
    # Initialize
    class_f1_scores = {}
    class_weights = {}
    
    # Count instances of each class in true labels
    total_samples = len(y_true)
    class_counts = Counter(y_true)
    
    # Calculate weights for each class
    for cls in classes:
        class_weights[cls] = class_counts.get(cls, 0) / total_samples
    
    # For each class, calculate F1 score
    for cls in classes:
        # True positives, false positives, false negatives
        tp = np.sum((y_true == cls) & (y_pred == cls) & ~((y_true == 0) & (y_pred == 0)))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        
        # Precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1 score for this class
        if precision + recall > 0:
            class_f1_scores[cls] = 2 * (precision * recall) / (precision + recall)
        else:
            class_f1_scores[cls] = 0
    
    # Calculate weighted F1 score
    weighted_f1 = sum(class_weights[cls] * class_f1_scores[cls] for cls in classes)
    return weighted_f1