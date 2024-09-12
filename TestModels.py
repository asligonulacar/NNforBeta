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
from tensorflow.keras.models import load_model
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
# Function to plot percentage confusion matrix
def plot_percentage_matrix(percentage_matrix, class_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(percentage_matrix, annot=True, fmt=".2f", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Percentage Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()

# Function to plot detailed confusion matrix
def plot_confusion_matrix(conf_matrix, class_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()

# Prepare data to test models.
colnames = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'bogus', 'negative', 'positive']
df = pd.read_csv("mlp_training_sample.csv", usecols=[i for i in range(9)], names = colnames, skiprows=0, sep=",", header=None, skip_blank_lines=True)
df = df.sample(frac=1).reset_index(drop=True)

# Split into train and test
train_frac = 0.6

test_arr = df.iloc[int(train_frac*len(df)):, 0:]

test_x = test_arr.iloc[:, :-3] 
test_y = test_arr.iloc[:, -3:]

test_x = test_x.to_numpy()
test_y = test_y.to_numpy()

test_dataset = tf.data.Dataset.from_tensor_slices((test_x,test_y))
test_dataset = test_dataset.batch(1)


model_paths = ['CAL1.h5', 'CAL2.h5', 'CAL3.h5', 'CAL4.h5', 'CAL5.h5', 'CAL6.h5']
models = {}
# Get model: Saved as CAL through CAL6 from the NNmodel.py file.
for i, path in enumerate(model_paths):
    model = load_model(path)  # Load the model from the file
    models[f'model_{i+1}'] = model  # Store the model in a dictionary

predictions = {}

for model_name, model in models.items():
    prediction = model.predict(test_x)
    # Get 0 or 1 by thresholding: set the max of the three entries to 1 and the others to zero. Softmax outputs a vector of probability scores.
    norm_pred = np.where(prediction == prediction.max(axis=1)[:, np.newaxis], 1, 0)
    predictions[model_name] = np.argmax(norm_pred, axis=1)
    print(f"Prediction from {model_name}: {predictions[model_name]}")

# Flatten the matrices based on your mapping
y_true = np.argmax(test_y, axis=1)

# Class names for the example based on your mapping
class_names = ['Bogus', 'Negative', 'Positive']


# Calculate the confusion matrix
for model_name, prediction in predictions.items():
    conf_matrix = confusion_matrix(y_true, prediction)
    print(conf_matrix)
    total_samples = np.sum(conf_matrix, axis=1, keepdims=True)
    # Calculate percentages
    percentage_matrix = conf_matrix / total_samples * 100  # Convert to percentage
    # Print the percentage matrix
    print("Percentage Matrix:\n", percentage_matrix)
    # Plot the percentage matrix
    plot_percentage_matrix(percentage_matrix, class_names)
    # Plot the confusion matrix
    plot_confusion_matrix(conf_matrix, class_names)




