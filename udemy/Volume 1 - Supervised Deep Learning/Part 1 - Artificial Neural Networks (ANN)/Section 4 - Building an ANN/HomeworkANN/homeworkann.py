import numpy as np
import pandas as pd

dataset = pd.read_csv("heart.csv")
#dataset = dataset.iloc[:300,:]
x = dataset.iloc[:,0:13].values
y = dataset.iloc[:,13].values



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
sexEncoder = OneHotEncoder(categorical_features = [1])
cpEncoder = OneHotEncoder(categorical_features = [2])
fbsEncoder = OneHotEncoder(categorical_features = [5])
restecgEncoder = OneHotEncoder(categorical_features = [6])
exangEncoder = OneHotEncoder(categorical_features = [8])
slopeEncoder = OneHotEncoder(categorical_features = [10])
caEncoder = OneHotEncoder(categorical_features = [11])
thalEncoder = OneHotEncoder(categorical_features = [12])
x = sexEncoder.fit_transform(x).toarray()
x = cpEncoder.fit_transform(x).toarray()
x = fbsEncoder.fit_transform(x).toarray()
x = restecgEncoder.fit_transform(x).toarray()
x = exangEncoder.fit_transform(x).toarray()
x = slopeEncoder.fit_transform(x).toarray()
x = caEncoder.fit_transform(x).toarray()
x = thalEncoder.fit_transform(x).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

import keras 
from keras.layers import Dense
from keras.models import Sequential

classifier = Sequential()

classifier.add(Dense(output_dim=8,activation="relu",init="uniform",input_dim=60))

classifier.add(Dense(output_dim=8,activation="relu",init="uniform"))

classifier.add(Dense(output_dim=1,activation="sigmoid",init="uniform"))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the Training Set
classifier.fit(x_train,y_train, batch_size = 10,nb_epoch = 100)

y_pred = classifier.predict(x_test)
y_pred_acc = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_acc)
















