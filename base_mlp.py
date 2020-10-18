from sklearn.neural_network import MLPClassifier
from evaluation import clf_evaluation


def run_base_mlp(ds_id, all_labels, train_features, train_labels, val_features, val_labels, test_features, test_labels):

    clf_base_mlp = MLPClassifier(hidden_layer_sizes=100, activation='logistic', solver='sgd')

    # Train
    clf_base_mlp.fit(train_features, train_labels)

    if val_features is not None and val_labels is not None:
        # Validation
        pred_labels = clf_base_mlp.predict(val_features)

        # Evaluation of Validation data
        clf_evaluation(val_labels, pred_labels, all_labels, 'Base-MLP_val.csv')
    else:
        # Test
        pred_labels = clf_base_mlp.predict(test_features)

        # Evaluation of Test data
        clf_evaluation(test_labels, pred_labels, all_labels, 'Base-MLP-' + ds_id + '.csv')
