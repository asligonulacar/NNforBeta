import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras
from keras import layers 
from keras.models import Sequential
import keras.optimizers
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Macro to create different neural network architectures. 
colnames = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'bogus', 'negative', 'positive']
df = pd.read_csv("mlp_training_sample.csv", usecols=[i for i in range(9)], names = colnames, skiprows=0, sep=",", header=None, skip_blank_lines=True)
df = df.sample(frac=1).reset_index(drop=True)
train_frac = 0.7

train_arr = df.iloc[:int(train_frac*len(df)), 0:]
test_arr = df.iloc[int(train_frac*len(df)):, 0:]

train_x = train_arr.iloc[:, :-3] 
train_y = train_arr.iloc[:, -3:]

test_x = test_arr.iloc[:, :-3] 
test_y = test_arr.iloc[:, -3:]

train_x = train_x.to_numpy()
train_y = train_y.to_numpy()
test_x = test_x.to_numpy()
test_y = test_y.to_numpy()

train_dataset = tf.data.Dataset.from_tensor_slices((train_x,train_y))
test_dataset = tf.data.Dataset.from_tensor_slices((test_x,test_y))

train_dataset = train_dataset.batch(64)
test_dataset = test_dataset.batch(1)
#tf.data.experimental.AUTOTUNE: I don't know what this does.
train_input_shape = train_x[0].shape

# I could also just have negative and positive as features, and if its not either its automatically classified as bogus.
# define deep learning model. 
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=train_input_shape),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(9, activation='relu'),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(3, activation='sigmoid')])

model2 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=train_input_shape),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='sigmoid')])

model3 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=train_input_shape),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(3, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(3, activation='sigmoid')])

model4 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=train_input_shape),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(3, activation='sigmoid')])

model5 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=train_input_shape),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')])

model6 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=train_input_shape),
    tf.keras.layers.Dense(6, activation='relu'),     
    tf.keras.layers.Dense(7, activation='relu'),    
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')])

model6.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model6.fit(train_dataset, epochs=125, validation_data=test_dataset)
model6.save('CAL6.h5')
