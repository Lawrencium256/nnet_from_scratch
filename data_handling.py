"""
------------------------------------------------------------------------------------------------------------------------
File to import and process MNIST data.
------------------------------------------------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_filepath = '/Users/lawrenceatkins/Desktop/nnet_from_scratch/mnist_data/mnist_train.csv'
test_filepath = '/Users/lawrenceatkins/Desktop/nnet_from_scratch/mnist_data/mnist_test.csv'

#Read CSVs as dataframe.
train_df = pd.read_csv(train_filepath)
test_df = pd.read_csv(test_filepath)

#Turn dataframes into numpy arrays.
train_data = train_df.iloc[:,1:]
train_data = train_data.to_numpy()
train_labels = train_df.iloc[:,0]
train_labels = train_labels.to_numpy()

test_data = test_df.iloc[:,1:]
test_data = test_data.to_numpy()
test_labels = test_df.iloc[:,0]
test_labels = test_labels.to_numpy()

#Renormalise.
test_data = test_data/255
train_data = train_data/255