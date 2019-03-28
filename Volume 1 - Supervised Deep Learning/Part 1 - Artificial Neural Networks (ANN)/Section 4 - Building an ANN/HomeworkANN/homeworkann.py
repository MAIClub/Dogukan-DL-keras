import numpy as np
import pandas as pd

dataset = pd.read_csv("Iris.csv")
x = dataset.iloc[:,1:4].values
y = dataset.iloc[:,-1].values

#%%
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit(x_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim = 2, init="uniform", activation="relu",input_dim=4))

classifier.add(Dense(output_dim = 2, init="uniform", activation="relu"))

classifier.add(Dense(output_dim = 1 ,init="uniform", activation))







