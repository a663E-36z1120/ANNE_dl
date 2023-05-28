import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

# Replace this example proportion-based confusion matrix with your own 3x3 matrix
confusion_mat_proportions = np.array([
    [0.75, 0.14, 0.11],
    [0.053, 0.51, 0.44],
    [0.02, 0.51, 0.47]
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
