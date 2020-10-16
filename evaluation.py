from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score
from csv_io import write_csv_data
import numpy as np


def clf_evaluation(true_label, pred_label, all_labels, filename):

    instance_pred = np.empty([pred_label.shape[0], 2], dtype=int)

    for index in range(pred_label.shape[0]):
        instance_pred[index] = [index + 1, pred_label[index]]

    confusion = confusion_matrix(true_label, pred_label, labels=all_labels)

    precision_by_classes = precision_score(true_label, pred_label, labels=all_labels, average=None, zero_division=0)

    recall_by_classes = recall_score(true_label, pred_label, labels=all_labels, average=None, zero_division=0)

    f1_measure_by_classes = f1_score(true_label, pred_label, labels=all_labels, average=None, zero_division=0)

    accuracy = accuracy_score(true_label, pred_label)

    macro_avg_f1 = f1_score(true_label, pred_label, labels=all_labels, average='macro', zero_division=0)

    weighted_avg_f1 = f1_score(true_label, pred_label, labels=all_labels, average='weighted', zero_division=0)

    write_csv_data(instance_pred, confusion, precision_by_classes, recall_by_classes, f1_measure_by_classes, accuracy,
                   macro_avg_f1, weighted_avg_f1, filename)
