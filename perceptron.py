from sklearn.linear_model import Perceptron
from csv_io import get_csv_data
from evaluation import clf_evaluation
import numpy as np


def run_perceptron(all_labels, train_features, train_labels, val_features, val_labels, test_features, test_labels):

    perceptron_clf = Perceptron()

    # Train
    perceptron_clf.fit(train_features, train_labels)

    # Validation
    # pred_labels = perceptron_clf.predict(val_features)

    # Evaluation of Validation data
    # clf_evaluation(val_labels, pred_labels, all_labels, 'PER-DS1.csv')

    # Test
    pred_labels = perceptron_clf.predict(test_features)

    # Evaluation of Test data
    clf_evaluation(test_labels, pred_labels, all_labels, 'PER-DS1.csv')
