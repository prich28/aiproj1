from gaus_naive_bayes import run_naive_bayes
from base_mlp import run_base_mlp
from best_mlp import run_best_mlp
from base_decision_tree import run_base_dt
from best_decision_tree import run_best_dt
from csv_io import get_csv_data
from perceptron import run_perceptron
import numpy as np
import matplotlib.pyplot as plt


ds_id = input('Provide an identifier for the dataset: ')
print(ds_id + ' will be used in the filenames')
print()

info_filepath = input('Enter filename of the class Info file (with extension): ')
info_filepath = 'dataset/' + info_filepath
print("Info file: " + info_filepath)
print()

class_number = input('Enter the number of classes in the dataset: ')
print('Number of classes indicated: ' + class_number)
number_of_classes = int(class_number)
print()

train_filepath = input('Enter filename of the training file (with extension): ')
train_filepath = 'dataset/' + train_filepath
print("Training file: " + train_filepath)
print()

is_validation = input('Type "V" to proceed with validation, otherwise leave blank for testing: ')
if is_validation:
    is_validation = True

else:
    is_validation = False
print()

validation_filepath = ''
test_label_filepath = ''
if is_validation:
    validation_filepath = input('Enter filename of the validation file (with extension): ')
    validation_filepath = 'dataset/' + validation_filepath
    print("Validation file: " + validation_filepath)
else:
    test_label_filepath = input('Enter filename of the test file with labels (with extension): ')
    test_label_filepath = 'dataset/' + test_label_filepath
    print("Test file: " + test_label_filepath)

print()

try:
    info = np.genfromtxt(info_filepath, skip_header=1, dtype='int', delimiter=',')
    number_of_classes = info.shape[0]
    print('Discovered ' + str(number_of_classes) + ' classes in the info file')
except:
    print()
    print('An error occurred importing your info file: ')
    print('Make sure the filename is spelled correctly')
    print('Make sure the first line is a label (and not data)')
    print()
    print('We will use the provided class number for now ' + number_of_classes)


# Training Data
train_features, train_labels = get_csv_data(train_filepath)
all_labels = np.arange(number_of_classes)

val_features = None
val_labels = None
test_features = None
test_labels = None
if is_validation:
    # Validation Data
    val_features, val_labels = get_csv_data(validation_filepath)
else:
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
run_naive_bayes(ds_id, all_labels, train_features, train_labels, val_features, val_labels, test_features, test_labels)

# Base Decision Tree
run_base_dt(ds_id, all_labels, train_features, train_labels, val_features, val_labels, test_features, test_labels)

# Best Decision Tree
run_best_dt(ds_id, all_labels, train_features, train_labels, val_features, val_labels, test_features, test_labels)

# Perceptron
run_perceptron(ds_id, all_labels, train_features, train_labels, val_features, val_labels, test_features, test_labels)

# Base-MLP
run_base_mlp(ds_id, all_labels, train_features, train_labels, val_features, val_labels, test_features, test_labels)

# Best-MLP
run_best_mlp(ds_id, all_labels, train_features, train_labels, val_features, val_labels, test_features, test_labels)
