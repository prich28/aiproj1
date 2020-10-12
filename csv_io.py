import numpy as np


def get_csv_data(file_name, has_labels):
    csv = np.genfromtxt(file_name, dtype='int', delimiter=",")

    num_columns = np.size(csv, 1)
    # print("total number of columns in csv: %s" % num_columns)

    # num_rows = np.size(csv, 0)

    if not has_labels:
        num_features = num_columns
        features = np.delete(csv, num_columns - 1, 1)
        return num_features, features
    else:
        num_features = num_columns - 1
        labels = csv[:, num_columns - 1]
        features = np.delete(csv, num_columns - 1, 1)
        # print("Label array: %s" % labels)
        # print("Features array: %s" % features)
        return num_features, features, labels


def write_csv_data(instance_pred, confusion, precision_by_classes, recall_by_classes, f1_measure_by_classes, accuracy,
                   macro_avg_f1, weighted_avg_f1):
    with open('GNB-DS1.csv', 'w') as f:
        f.write('Predictions\n')
        np.savetxt(f, instance_pred, delimiter=',', fmt='%i')
        f.write('\n')

        f.write('Confusion Matrix\n')
        np.savetxt(f, confusion, delimiter=',', fmt='%i')
        f.write('\n')

        f.write('Precision by class\n')
        np.savetxt(f, precision_by_classes, delimiter=',', fmt='%f')
        f.write('\n')

        f.write('Recall by class\n')
        np.savetxt(f, recall_by_classes, delimiter=',', fmt='%f')
        f.write('\n')

        f.write('F1 by class\n')
        np.savetxt(f, f1_measure_by_classes, delimiter=',', fmt='%f')
        f.write('\n')

        f.write('Accuracy\n')
        f.write(str(accuracy))
        f.write('\n')
        f.write('\n')

        f.write('Macro F1 Average\n')
        f.write(str(macro_avg_f1))
        f.write('\n')
        f.write('\n')

        f.write('Weighted F1 Average\n')
        f.write(str(weighted_avg_f1))
        f.write('\n')
