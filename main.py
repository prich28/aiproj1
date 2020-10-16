import os

from gaus_naive_bayes import run_naive_bayes
from base_mlp import run_base_mlp
from base_decision_tree import run_base_dt
from best_decision_tree import run_best_dt
from csv_io import get_csv_data
from perceptron import run_perceptron
import numpy as np
import matplotlib.pyplot as plt


info_filepath = input('Enter filename of the class Info file (with extension): ')
info_filepath = 'dataset/' + info_filepath
print("Info file: " + info_filepath)

train_filepath = input('Enter filename of the training file (with extension): ')
train_filepath = 'dataset/' + train_filepath
print("Training file: " + train_filepath)

validation_filepath = input('Enter filename of the validation file (with extension): ')
validation_filepath = 'dataset/' + validation_filepath
print("Validation file: " + validation_filepath)

test_label_filepath = input('Enter filename of the test file with labels (with extension): ')
test_label_filepath = 'dataset/' + test_label_filepath
print("Validation file: " + test_label_filepath)

info = np.genfromtxt(info_filepath, skip_header=1, dtype='int', delimiter=',')
number_of_classes = info.shape[0]
print(number_of_classes)

# Training Data
train_features, train_labels = get_csv_data(train_filepath)
all_labels = np.arange(number_of_classes)

# Validation Data
val_features, val_labels = get_csv_data(validation_filepath)

# Testing Data
test_features, test_labels = get_csv_data(test_label_filepath)

# Plot Class distribution of training data
unique, counts = np.unique(train_labels, return_counts=True)
plt.bar(unique, counts)

plt.title('Class Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')

plt.show()

# Gaussian Naive Bayes
run_naive_bayes(all_labels, train_features, train_labels, val_features, val_labels, test_features, test_labels)

# Base Decision Tree
run_base_dt(all_labels, train_features, train_labels, val_features, val_labels, test_features, test_labels)

# Best Decision Tree
run_best_dt(all_labels, train_features, train_labels, val_features, val_labels, test_features, test_labels)

# Perceptron
run_perceptron(all_labels, train_features, train_labels, val_features, val_labels, test_features, test_labels)

# Base-MLP
run_base_mlp(all_labels, train_features, train_labels, val_features, val_labels, test_features, test_labels)

# Best-MLP
