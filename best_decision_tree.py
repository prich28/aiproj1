from sklearn.tree import DecisionTreeClassifier
from evaluation import clf_evaluation
from sklearn.model_selection import GridSearchCV


def run_best_dt(ds_id, all_labels, train_features, train_labels, val_features, val_labels, test_features, test_labels):

    # These hyper parameters were found using the GridSearchCV algorithm
    base_dt_clf = DecisionTreeClassifier(criterion='entropy', max_depth=None, class_weight=None,
                                         min_impurity_decrease=0, min_samples_split=2)

    # Define hyper params to search
    # parameter_space = {
    #     'criterion': ['gini', 'entropy'],
    #     'max_depth': [10, None],
    #     'min_samples_split': [8, 10, 20, 30, 40],
    #     'min_impurity_decrease': [0, 0.0002, 0.0003, 0.0001],
    #     'class_weight': [None, 'balanced']
    # }

    # dt_clf_with_gs = GridSearchCV(base_dt_clf, parameter_space, n_jobs=-1, cv=5)

    # Train
    base_dt_clf = base_dt_clf.fit(train_features, train_labels)

    # print('Best parameters found:\n', dt_clf_with_gs.best_params_)

    if val_features is not None and val_labels is not None:
        # Validation
        pred_labels = base_dt_clf.predict(val_features)

        # Evaluation of Validation data
        clf_evaluation(val_labels, pred_labels, all_labels, 'Best-DT_val.csv')
    else:
        # Test
        pred_labels = base_dt_clf.predict(test_features)

        # Evaluation of Test data
        clf_evaluation(test_labels, pred_labels, all_labels, 'Best-DT-' + ds_id + '.csv')
