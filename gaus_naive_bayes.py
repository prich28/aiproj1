from evaluation import clf_evaluation
from sklearn.naive_bayes import GaussianNB


def run_naive_bayes(ds_id, all_labels, train_features, train_labels, val_features, val_labels, test_features, test_labels):

    clf = GaussianNB()

    # Train
    clf.fit(train_features, train_labels)

    if val_features is not None and val_labels is not None:
        # Validation
        pred_labels = clf.predict(val_features)

        # Evaluation of Validation data
        clf_evaluation(val_labels, pred_labels, all_labels, 'GNB_val.csv')
    else:
        # Test
        pred_labels = clf.predict(test_features)

        # Evaluation of Test data
        clf_evaluation(test_labels, pred_labels, all_labels, 'GNB-' + ds_id + '.csv')
