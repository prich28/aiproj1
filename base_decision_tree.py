from csv_io import get_csv_data
from sklearn.tree import DecisionTreeClassifier
from evaluation import clf_evaluation
import numpy as np


def run_base_dt(all_labels, train_features, train_labels, val_features, val_labels, test_features, test_labels):

    base_dt_clf = DecisionTreeClassifier(criterion='entropy')

    # Train
    base_dt_clf = base_dt_clf.fit(train_features, train_labels)

    # Validation
    # num_val_features, val_features, val_labels = get_csv_data(validation_file, True)
    # pred_labels = base_dt_clf.predict(val_features)

    # Evaluation of Validation data
    # clf_evaluation(val_labels, pred_labels, all_labels, 'Base-DT-DS1.csv')

    # Test
    pred_labels = base_dt_clf.predict(test_features)

    # Evaluation of Test data
    clf_evaluation(test_labels, pred_labels, all_labels, 'Base-DT-DS1.csv')
