from evaluation import clf_evaluation
from sklearn.naive_bayes import GaussianNB


def run_naive_bayes(all_labels, train_features, train_labels, val_features, val_labels, test_features, test_labels):

    clf = GaussianNB()

    # Train
    clf.fit(train_features, train_labels)

    # Validation
    # pred_labels = clf.predict(val_features)

    # Evaluation of Validation data
    # clf_evaluation(val_labels, pred_labels, all_labels, 'GNB-DS1.csv')

    # Test
    pred_labels = clf.predict(test_features)

    # Evaluation of Test data
    clf_evaluation(test_labels, pred_labels, all_labels, 'GNB-DS1.csv')
