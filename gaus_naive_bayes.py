from csv_io import get_csv_data
from evaluation import clf_evaluation
from sklearn.naive_bayes import GaussianNB

import numpy as np


def run_naive_bayes(train_file, validation_file, test_label_file):
    num_features, features, labels = get_csv_data(train_file, True)

    # TODO import classes from file put in main
    all_labels = np.arange(26)
    # print(all_labels)

    clf = GaussianNB()

    # Train
    clf.fit(features, labels)
    print(clf.class_prior_)

    # Validation
    num_val_features, val_features, val_labels = get_csv_data(validation_file, True)
    pred_labels = clf.predict(val_features)
    # print(val_labels)
    # print(pred_labels)
    #
    clf_evaluation(val_labels, pred_labels, all_labels)

    # Test
    # num_test_features, test_features, test_labels = get_csv_data(test_label_file, True)
    # print(clf.predict(test_features))
