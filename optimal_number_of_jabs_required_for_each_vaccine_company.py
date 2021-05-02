# Importing the libraries
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import time

# Getting the dataset
X = pd.read_csv("2021VAERSDATA.csv").iloc[:5000, 0: 18].values
Y = pd.read_csv("2021VAERSVAX.csv").iloc[:5000, 0: 5].values

# Ultimate Array (Initialization)
total_data = []

# Matching the patients with the paramaters
for i in range(0, len(X)):  # Length of X as X is shorter
    primary_key_X = X[i, 0]
    selected_data = []
    for j in range(0, len(Y)):
        primary_key_Y = Y[j, 0]
        if primary_key_X == primary_key_Y:
            selected_data = [primary_key_Y, Y[j, 2], Y[j, 4], X[i, 17]]
    total_data.append(selected_data)

# # Preprocessing the input dataset
# total_data = np.array(total_data, dtype=object)

# Splitting independant to dependant
indep_value = []
dep_value = []
for i in range(0, 4930):  # len(total_data)):
    curr_element = total_data[i]

    if len(curr_element) <= 3:
        continue
    else:
        indep_value.append(curr_element[3])
        dep_value.append(curr_element[0:3])


# Applying the Machine Learning Model
indep_value = np.array(indep_value).reshape(-1, 1)
dep_value = np.array(dep_value)
training_set = np.concatenate((dep_value, indep_value), axis=1)

# Further Preprocessing of data

# Removing the nan and U
new_training_set = []
for setv in training_set:
    if setv[3] == "Y" or setv[3] == "N":
        new_training_set.append(setv)
    else:
        continue
new_training_set = np.array(new_training_set)
training_set = new_training_set

# Processing the Vaccine Types
vac_type_le = LabelEncoder()
training_set[:, 1] = vac_type_le.fit_transform(training_set[:, 1])
re_covid_le = LabelEncoder()
training_set[:, 3] = re_covid_le.fit_transform(training_set[:, 3])
training_set = np.array(training_set)
le = LabelEncoder()
training_set[:, 2] = le.fit_transform(training_set[:, 2])
# Removing the patient ID
training_set = training_set[:, 1:]

# Extracting the labels
vac_type_label_mapping = dict(
    zip(vac_type_le.classes_, vac_type_le.transform(vac_type_le.classes_)))
re_covid_label_mapping = dict(
    zip(re_covid_le.classes_, re_covid_le.transform(re_covid_le.classes_)))
le_mapping = dict(
    zip(le.classes_, le.transform(le.classes_)))
# Converting values to dep and indep
dep_training = training_set[:, 1].reshape(-1, 1)  # !Y
indep_training = np.concatenate(
    (training_set[:, 0].reshape(-1, 1), training_set[:, 2].reshape(-1, 1)), axis=1)  # !X

# Scaling the indep data

sc = StandardScaler()
sc.fit_transform(indep_training)
print(indep_training.shape)
# Applying the Machine Learning model
nn = Sequential()
# input_shape=(indep_training.shape)))
nn.add(Dense(units=120, activation='relu'))
nn.add(Dense(units=60, activation='relu'))
# nn.add(Dense(units=15, activation='relu'))
nn.add(Dense(units=1))
nn.compile(optimizer="adam", loss="mse",
           metrics=[tf.keras.metrics.MeanSquaredError()])
nn.fit(indep_training, dep_training, batch_size=100, epochs=4000)

# # Printing the Labels
print(le_mapping)
print(vac_type_label_mapping)
print(re_covid_label_mapping)


#! Moderna (2)
moderna_optim = nn.predict_classes(np.array(([2, 0], [2, 1])))
print(moderna_optim)

#! Pfizer (2)
pfizer_optim = nn.predict_classes(np.array(([3, 0], [3, 1])))
print(pfizer_optim)
