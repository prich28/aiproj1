from gaus_naive_bayes import run_naive_bayes

train_file = './dataset/train/train_1.csv'
validation_file = './dataset/validate/val_1.csv'
test_label_file = './dataset/test/test_with_label_1.csv'
test_no_label_file = './dataset/test/test_no_label_1.csv'

# Gaussian Naive Bayes
run_naive_bayes(train_file, validation_file, test_label_file)

