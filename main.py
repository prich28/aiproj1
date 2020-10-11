from csv_io import get_csv_data
import numpy as np
from sklearn.naive_bayes import GaussianNB

train_file = './dataset/train/train_1.csv'

num_features, features, labels = get_csv_data()

clf = GaussianNB()
clf.fit(features, labels)

