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
# Prepare data to test models.
colnames = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'var1', 'var2', 'var3']
df = pd.read_csv("train_cf_onlyFTOF.csv", usecols=[i for i in range(9)], names = colnames, skiprows=0, sep=",", header=None, skip_blank_lines=True)
df = df.sample(frac=1).reset_index(drop=True)

df_var1 = df.copy()
df_var2 = df.copy()
df_var3 = df.copy()

df_var1.drop(df_var1.columns[[7,8]], axis=1, inplace=True)
df_var2.drop(df_var2.columns[[6,8]], axis=1, inplace=True)
df_var3.drop(df_var3.columns[[6,7]], axis=1, inplace=True)
# Split into train and test


train_frac = 0.7

test_arr = df.iloc[int(train_frac*len(df)):, 0:]
test_arr_var1 = df_var1.iloc[int(train_frac*len(df_var1)):, 0:]
test_arr_var2 = df_var2.iloc[int(train_frac*len(df_var2)):, 0:]
test_arr_var3 = df_var3.iloc[int(train_frac*len(df_var3)):, 0:]

test_x = test_arr_var1.iloc[:, :-1]
test_xall = test_arr.iloc[:, :-3]


test_y = test_arr.iloc[:, -3:]
test_y_var1 = test_arr_var1.iloc[:, -1:]
test_y_var2 = test_arr_var2.iloc[:, -1:]
test_y_var3 = test_arr_var3.iloc[:, -1:]
test_x_np = test_x.to_numpy()
test_xall_np = test_xall.to_numpy()


test_y_np =  test_y.to_numpy()
test_y_var1_np = test_y_var1.to_numpy()
test_y_var2_np = test_y_var2.to_numpy()
test_y_var3_np = test_y_var3.to_numpy()



test_dataset = tf.data.Dataset.from_tensor_slices((test_xall_np, test_y_np))
test_dataset_var1 = tf.data.Dataset.from_tensor_slices((test_x_np,test_y_var1_np))
test_dataset_var2 = tf.data.Dataset.from_tensor_slices((test_x_np,test_y_var2_np))
test_dataset_var3 = tf.data.Dataset.from_tensor_slices((test_x_np,test_y_var3_np))

test_dataset = test_dataset.batch(1)
test_dataset_var1 = test_dataset_var1.batch(1)
test_dataset_var2 = test_dataset_var2.batch(1)
test_dataset_var3 = test_dataset_var3.batch(1)


# Get model.
model = tf.keras.models.load_model("RegOneForAll.h5")
model1 = tf.keras.models.load_model("RegOneForVar1.h5")
model2 = tf.keras.models.load_model("RegOneForVar2.h5")
model3 = tf.keras.models.load_model("RegOneForVar3.h5")
# Get predictions 
ypred = model.predict(test_xall_np)
ypred_var1 = model1.predict(test_x_np)
ypred_var2 = model2.predict(test_x_np)
ypred_var3 = model3.predict(test_x_np)

print(ypred_var3)
# Flatten the matrices 
y_true = np.ndarray.flatten(test_y_np)
y_pred = np.ndarray.flatten(ypred)

y_true1 = np.ndarray.flatten(test_y_var1_np)
y_pred1 = np.ndarray.flatten(ypred_var1)

y_true2 = np.ndarray.flatten(test_y_var2_np)
y_pred2 = np.ndarray.flatten(ypred_var2)


y_true3 = np.ndarray.flatten(test_y_var3_np)
y_pred3 = np.ndarray.flatten(ypred_var3)

# Calculate metrics.
for i in range(0,3):
    
    loss_ofa = np.mean((test_y_np[:,i] - ypred[:,i])**2)
    sum_squares_residuals_ofa = sum((test_y_np[:,i] - ypred[:,i]) ** 2)
    sum_squares_ofa = sum((test_y_np[:,i] - np.mean(test_y_np[:,i])) ** 2)
    R2_ofa = 1 - sum_squares_residuals_ofa / sum_squares_ofa
    mae = np.mean(abs(test_y_np[:,i] - ypred[:,i]))
    print("MSE of var %d with VarAll: " %(i+1), loss_ofa)
    print("R2 of var %d with VarAll:" %(i+1), R2_ofa)
    print("MAE of var %d with VarAll: " %(i+1), mae)

sum_squares_residuals_var1 = sum((y_true1 - y_pred1) ** 2)
sum_squares_var1 = sum((y_true1 - np.mean(y_true1)) ** 2)
loss_var1 = np.mean((y_true1 - y_pred1)**2)
R2_var1 = 1 - sum_squares_residuals_var1 / sum_squares_var1
mae_var1 = np.mean(abs(y_true1 - y_pred1))

print("MSE of var 1 with Var1:", loss_var1)
print("R2 of var 1 with Var1:", R2_var1)
print("MAE of var 1 with Var1:", mae_var1)

loss_var2 = np.mean((y_true2 - y_pred2)**2)
sum_squares_residuals_var2 = sum((y_true2 - y_pred2) ** 2)
sum_squares_var2 = sum((y_true2 - np.mean(y_true2)) ** 2)
R2_var2 = 1 - sum_squares_residuals_var2 / sum_squares_var2
mae_var2 = np.mean(abs(y_true2 - y_pred2))

print("MSE of var 2 with Var2:",loss_var2)
print("R2 of var 2 with Var2:", R2_var2)
print("MAE of var 2 with Var2:", mae_var2)

loss_var3 = np.mean((y_true3 - y_pred3)**2)
sum_squares_residuals_var3 = sum((y_true3 - y_pred3) ** 2)
sum_squares_var3 = sum((y_true3 - np.mean(y_true3)) ** 2)
R2_var3 = 1 - sum_squares_residuals_var3 / sum_squares_var3
mae_var3 = np.mean(abs(y_true3 - y_pred3))

print("MSE of var 3 with Var3:", loss_var3)
print("R2 of var 3 with Var3:", R2_var3)
print("MAE of var 3 with Var3:", mae_var3)
