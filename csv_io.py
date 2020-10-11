import numpy as np


def get_csv_data(file_name):
    csv = np.genfromtxt(file_name, delimiter=",")

    num_columns = np.size(csv, 1)
    # print("total number of columns in csv: %s" % num_columns)

    # num_rows = np.size(csv, 0)

    num_features = num_columns - 1

    labels = csv[:, num_columns - 1]
    features = np.delete(csv, num_columns - 1, 1)

    # print("Label array: %s" % labels)
    # print("Features array: %s" % features)

    return num_features, features, labels
