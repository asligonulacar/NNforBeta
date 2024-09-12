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

# Macro to create regression NN models. 
colnames = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'var1', 'var2', 'var3']
df = pd.read_csv("train_cf_onlyFTOF.csv", usecols=[i for i in range(9)], names = colnames, skiprows=0, sep=",", header=None, skip_blank_lines=True)
df = df.sample(frac=1).reset_index(drop=True)
df.drop(df.columns[[7,8]], axis=1, inplace=True)
print(df.head(5))
train_frac = 0.7

train_arr = df.iloc[:int(train_frac*len(df)), 0:]
test_arr = df.iloc[int(train_frac*len(df)):, 0:]

# Fix this to fit different shapes of df.
train_x = train_arr.iloc[:, :-1] 
train_y = train_arr.iloc[:, -1:]


test_x = test_arr.iloc[:, :-1] 
test_y = test_arr.iloc[:, -1:]
print(test_y.head(5))
print(test_x.head(5))
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

# Define deep learning model. 

model = tf.keras.Sequential([

    tf.keras.layers.InputLayer(input_shape=train_input_shape),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')])



model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.MeanSquaredError(reduction="sum_over_batch_size", name="mean_squared_error"), metrics=['mse', 'mae'])
history = model.fit(train_dataset, epochs=125, validation_data=test_dataset)
plt.plot(history.history['val_loss'])
plt.show()
model.evaluate(train_dataset)
model.save('RegOneForVar1.h5')
