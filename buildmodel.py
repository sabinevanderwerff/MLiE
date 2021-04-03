# -*- coding: utf-8 -*-
"""
Project: Make a model
Created on Sat Apr  3 15:58:49 2021

@author: Sabine
"""

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow import initializers

df = pd.read_pickle("data/mapped_data.pkl")
df.drop(['longitude','latitude'], axis=1, inplace=True)

# SELECT AND SHUFFLE DATASET
# First, select the first 5% for training and testing (always the same)
# Then, shuffle to input in randomized order
# CHANGE THIS ORDER FOR ACTUAL TRAINING!
y = df['rent']                  # the rent is our target output
x = df.drop('rent',axis=1)      # all other values are features
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x, y, test_size = 0.05, train_size = 0.05, shuffle = False)

xTrain, yTrain = shuffle(xTrain, yTrain) 
xTest, yTest = shuffle(xTest, yTest) 

# GENERATE MODEL
initializer = keras.initializers.HeNormal()     # Recommended for reLU function
model = keras.models.Sequential()
model.add(keras.layers.Dense(32, input_dim = xTrain.shape[1], activation='relu', kernel_initializer=initializer))


