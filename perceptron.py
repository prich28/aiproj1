from sklearn.linear_model import Perceptron
from evaluation import clf_evaluation


def run_perceptron(ds_id, all_labels, train_features, train_labels, val_features, val_labels, test_features, test_labels):

    perceptron_clf = Perceptron()

    # Train
    perceptron_clf.fit(train_features, train_labels)

    if val_features is not None and val_labels is not None:
        # Validation
        pred_labels = perceptron_clf.predict(val_features)

        # Evaluation of Validation data
        clf_evaluation(val_labels, pred_labels, all_labels, 'PER_val.csv')
    else:
        # Test
        pred_labels = perceptron_clf.predict(test_features)

        # Evaluation of Test data
        clf_evaluation(test_labels, pred_labels, all_labels, 'PER-' + ds_id + '.csv')
