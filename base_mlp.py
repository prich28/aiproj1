from sklearn.neural_network import MLPClassifier
from csv_io import get_csv_data
from evaluation import clf_evaluation
import numpy as np


def run_base_mlp(all_labels, train_features, train_labels, val_features, val_labels, test_features, test_labels):

    clf_base_mlp = MLPClassifier(hidden_layer_sizes=100, activation='logistic', solver='sgd')

    # Train
    clf_base_mlp.fit(train_features, train_labels)

    # Validation
    # num_val_features, val_features, val_labels = get_csv_data(validation_file, True)
    # pred_labels = clf_base_mlp.predict(val_features)

    # Evaluation of Validation data
    # clf_evaluation(val_labels, pred_labels, all_labels, 'Base-MLP-DS1.csv')

    # Test
    pred_labels = clf_base_mlp.predict(test_features)

    # Evaluation of Test data
    clf_evaluation(test_labels, pred_labels, all_labels, 'Base-MLP-DS1.csv')
