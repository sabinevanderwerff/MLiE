#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 16:00:43 2021

@author: eliannehopman
"""

import pandas as pd
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

plt.style.use('dark_background')

#---- Data set import and shuffle ----
df = pd.read_pickle("data/mapped_data_GGD_encoded.pkl")
df = df[df.rent < 2000]
location = df.iloc[:,17::].columns.values.tolist()  # list with all regions

# Remove maximum 10 samples
maxrent = df.rent.sort_values(ascending=False)
maxarea = df.area_sqm.sort_values(ascending=False)
#maxcost = df.additional_costs.sort_values(ascending=False)
maxdeposit = df.deposit.sort_values(ascending=False)

for i in range(10):
    df = df[df.rent!=maxrent.iloc[i]]
    df = df[df.area_sqm!=maxarea.iloc[i]]
    #df = df[df.additional_costs!=maxcost.iloc[i]]
    df = df[df.deposit!=maxdeposit.iloc[i]]

comparefeatures = 0   # 1 if we investigate similar inputs
investigatefeature = location # choose: location, ['area_sqm'], ['furnish'],  ['living'], (['internet'], ['kitchen'])
testperc = 0.10    # split data, default value
trainperc = 0.90   # split data, default value

# investigate similarity of certain features
if comparefeatures == 1:
    df = df.loc[df.property_type==4, :]    # ONLY INVESTIGATE ROOMS (otherwise issues with shared/own etc for other properties)

    # Drop all the not_significant features and feature that we want to investigate > their value does not matter for estimating rent
    # We need to fix all the features that áre significant and not under investigation > equal circumstances

    # Variables that might influence the rent significantly
    excludealways = ['rent']
    significantfeatures = ['area_sqm', 'furnish',  'internet', 'kitchen', 'living']

    # Variables that do not influence the rent signifcantly
    not_significantfeatures = ['roommates', 'deposit', 'additional_costs', 'pets', 'toilet', 'smoking_inside', 'shower', 'registration_cost', 'rent_detail', 'energy_label'] # parameters with low correlation to rent

    itemstodrop = not_significantfeatures + investigatefeature + excludealways

    print('Properties that are fixed:')
    for house_property in df.drop(itemstodrop, axis=1):
        select = df[house_property].unique()[0]         # select the property with most samples in it
        df = df.loc[df[house_property]==select, :]
        print(house_property, select)
    
    print('amount of samples: ', len(df))
    testperc = 0.5
    trainperc = 0.5
# end if

# Seperate features and target
x = df.loc[:, df.columns != 'rent']
y = df['rent']

# Taking first 10% of dataset and splitting into 5% test and 5% train
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=testperc, train_size=testperc, shuffle=False)
# Shuffeling data
xTrain, yTrain = shuffle(xTrain, yTrain)
xTest, yTest = shuffle(xTest, yTest)

#---- Model ----

# Building model
init = tf.keras.initializers.HeUniform() #Initializer recommended for ReLu (also try HeUniform)
act = 'relu' # Actfunc is same in each hidden layer

model = Sequential()
model.add(Dense(128, activation=act,kernel_initializer=init))
model.add(Dense(128, activation=act,kernel_initializer=init))
model.add(Dense(64, activation=act,kernel_initializer=init))
model.add(Dense(32, activation=act,kernel_initializer=init))
model.add(Dense(16, activation=act,kernel_initializer=init))
model.add(Dense(8, activation=act,kernel_initializer=init))
model.add(Dense(1, activation='linear')) # Output layer has linear actfunc

# Compiling model
eta = 0.001 # Default eta for 'adam', we can adjust this
opt = tf.keras.optimizers.Adam(learning_rate=eta)
model.compile(optimizer=opt,loss='mae',metrics=['mean_absolute_error'])

# Fitting model
eps = 100
batch = 32 # Default bath size
valsplit = 0.2 # Fraction of training data to be used as validation data at end of each epoch

fitting = model.fit(xTrain,yTrain,epochs=eps,batch_size=batch,verbose=0,validation_split=valsplit)
loss = fitting.history['mean_absolute_error']
lossnorm = [x/max(loss) for x in loss]

# evaluating model
error = model.evaluate(xTest,yTest, verbose=0) # Check even welke error dit is, laatste, laagste, gemiddelde???
print('MAE evaluation: %.4f \n\r RMAE evaluation: %.4f' % (error[0], np.sqrt(error[0])))

# predicting
pred = model.predict(xTest)


#---- Comparison predicted and actual rent price ----

# Plots
fig1, ax = plt.subplots(2)
fig1.suptitle('MLP for predicting rent prices')

ax[0].scatter(xTest.index,yTest,c='r',s=2,label='True Data', alpha=0.8)
ax[0].scatter(xTest.index,pred,c='b',s=2,label='Prediction by MLP', alpha=0.8)
ax[0].legend(loc="upper right", prop={'size':8})
ax[0].set_xlabel("Room index")
ax[0].set_ylabel("Rent price")
ax[0].set_title("MAE: %.4f, RMAE: %.4f" % (error[0], np.sqrt(error[0])),fontsize=10)

ax[1].semilogy(lossnorm, label='Mean Absolute Error',c='r')  #MSE
ax[1].legend(loc="upper right", prop={'size':8})
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("MAE (normalised)")
ax[1].grid(True, which='both')


# ---- Check variation ----

# Calculate the percentage of deviation of actual price
yTest = np.asarray(yTest).reshape((len(yTest),1))
perc = (pred-yTest)/yTest*100
bins = np.arange(-110,120,10)

# Plot histogram
fig2, ax = plt.subplots(1)
y, x, _ = ax.hist(np.clip(np.ravel(perc), bins[0], bins[-1]), bins, color='r')
for rect in ax.patches:
    height = rect.get_height()
    plt.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height),
                xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')
labels = ['-100+%', '-100%', '-90%', '-80%', '-70%', '-60%', '-50%', '-40%', '-30%', '-20%', '-10%', '0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%', '100+%']
ax.set_xlabel("Percentage of deviation")
ax.set_ylabel("Number of houses")
ax.set_title("Deviation in predicted price",fontsize=10)
ax.set_xticks(bins)
ax.set_xticklabels(labels, rotation='vertical')
ax.text(50, y.max()/2, " Min: %.2f%% \n Max: %.2f%% \n Mean: %.2f%% \n Std: %.2f%% \n Median: %.2f%%" % (min(perc), max(perc), np.average(perc), np.std(perc), np.median(perc)))
plt.show()

# --- Detect and analyse outliers ---

# if point is outside mean +- std, it should be classified as an outlier
cutoffhigh = np.average(perc) + np.std(perc)
cutofflow = np.average(perc) - np.std(perc)
perc = perc.reshape(1, len(perc))[0].tolist()
outliers = [x for x in perc if x > 100]

# These outliers are in perc, having the same index as pred/yTest/xTest
ind = [perc.index(x) for x in outliers]
features_outlier = xTest.iloc[ind,:].transpose()

features_outlier.to_pickle('data/features_outliers.pkl')




