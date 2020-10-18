from sklearn.tree import DecisionTreeClassifier
from evaluation import clf_evaluation


def run_base_dt(ds_id, all_labels, train_features, train_labels, val_features, val_labels, test_features, test_labels):

    base_dt_clf = DecisionTreeClassifier(criterion='entropy')

    # Train
    base_dt_clf = base_dt_clf.fit(train_features, train_labels)

    if val_features is not None and val_labels is not None:
        # Validation
        pred_labels = base_dt_clf.predict(val_features)

        # Evaluation of Validation data
        clf_evaluation(val_labels, pred_labels, all_labels, 'Base-DT_val.csv')
    else:
        # Test
        pred_labels = base_dt_clf.predict(test_features)

        # Evaluation of Test data
        clf_evaluation(test_labels, pred_labels, all_labels, 'Base-DT-' + ds_id + '.csv')
