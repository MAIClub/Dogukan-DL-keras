with open("googleplaystore.csv", "r") as f:
    lines = f.readlines()
with open("googleplaystore.csv", "w") as f:
    for line in lines:
        if "nan" not in line:
            f.write(line)

#%%
            
import pandas as pd
import numpy as np
## data cleanup ##

df = pd.read_csv("googleplaystore.csv")
del df["Last Updated"]
size = df.iloc[:,4].values
installs = df.iloc[:,5].values
for i in range(len(size)):
    size[i] = size[i].replace("M","")
    size[i] = size[i].replace("k","") 
for i in range(len(installs)):
    installs[i] = installs[i].replace(",","")

x = df.iloc[:,1:5].join(df.iloc[:,6:10]).values
y = df["Installs"].values
#Label and Hot Encode
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

categoryEncoder = LabelEncoder()
typeEncoder = LabelEncoder()
contentEncoder = LabelEncoder()
genreEncoder = LabelEncoder()
x[:,0] = categoryEncoder.fit_transform(x[:, 0])
x[:,4] = typeEncoder.fit_transform(x[:,4])
x[:,6] = contentEncoder.fit_transform(x[:,6])
x[:,7] = genreEncoder.fit_transform(x[:,7])
oneHotEncoder = OneHotEncoder(categorical_features=[0,4,6,7])
x = oneHotEncoder.fit_transform(x).toarray()
yencoder = LabelEncoder()
y =  yencoder.fit_transform(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state = 0)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from keras.models import Sequential
from keras.layers import Dense

#%%


model =Sequential()
model.add(Dense(units = 78, kernel_initializer = 'uniform', activation = 'softmax', input_dim = 155))
model.add(Dense(units = 78, kernel_initializer = 'uniform', activation = 'softmax'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 10, epochs = 100)










  





