import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

# Replace this example proportion-based confusion matrix with your own 3x3 matrix
confusion_mat_proportions = np.array([
    [0.79, 0.16, 0.049],
    [0.14, 0.61, 0.25],
    [0.15, 0.47, 0.38]
])

def f1_from_proportional_confusion_matrix(confusion_mat_proportions):
    true_positives = np.diag(confusion_mat_proportions)
    false_positives = np.sum(confusion_mat_proportions, axis=0) - true_positives
    false_negatives = np.sum(confusion_mat_proportions, axis=1) - true_positives

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    f1_scores = 2 * (precision * recall) / (precision + recall)

    return f1_scores

f1_scores = f1_from_proportional_confusion_matrix(confusion_mat_proportions)
avg_f1_score = np.mean(f1_scores)

print("F1 scores for each class:", f1_scores)
print("Average F1 score:", avg_f1_score)
