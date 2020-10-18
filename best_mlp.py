from sklearn.neural_network import MLPClassifier
from evaluation import clf_evaluation
from sklearn.model_selection import GridSearchCV


def run_best_mlp(ds_id, all_labels, train_features, train_labels, val_features, val_labels, test_features, test_labels):

    # These hyper parameters were found using the GridSearchCV algorithm
    best_mlp_clf = MLPClassifier(hidden_layer_sizes=(60, 80), activation='tanh', solver='adam')

    # Define hyper params to search
    # parameter_space = {
    #     'hidden_layer_sizes': [(50, 80), (20, 20, 20)],
    #     'activation': ['identity', 'tanh', 'logistic', 'relu'],
    #     'solver': ['adam', 'sgd']
    # }

    # mlp_clf_with_gs = GridSearchCV(best_mlp_clf, parameter_space, n_jobs=-1, cv=5)

    # Train
    best_mlp_clf = best_mlp_clf.fit(train_features, train_labels)

    # print('Best parameters found:\n', mlp_clf_with_gs.best_params_)

    if val_features is not None and val_labels is not None:
        # Validation
        pred_labels = best_mlp_clf.predict(val_features)

        # Evaluation of Validation data
        clf_evaluation(val_labels, pred_labels, all_labels, 'Best-MLP_val.csv')
    else:
        # Test
        pred_labels = best_mlp_clf.predict(test_features)

        # Evaluation of Test data
        clf_evaluation(test_labels, pred_labels, all_labels, 'Best-MLP-' + ds_id + '.csv')
