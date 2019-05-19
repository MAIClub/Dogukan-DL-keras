            
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
from sklearn.externals import joblib
from keras.utils import np_utils
categoryEncoder = LabelEncoder()
#ypeEncoder = LabelEncoder()
#contentEncoder = LabelEncoder()
#genreEncoder = LabelEncoder()
x[:,0] = categoryEncoder.fit_transform(x[:, 0])
x[:,4] = categoryEncoder.fit_transform(x[:,4])
x[:,6] = categoryEncoder.fit_transform(x[:,6])
x[:,7] = categoryEncoder.fit_transform(x[:,7])
oneHotEncoder = OneHotEncoder(categorical_features=[0,4,6,7])
x = oneHotEncoder.fit_transform(x).toarray()
joblib.dump(categoryEncoder,"encoder.pkl")

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,dummy_y, test_size=0.2,random_state = 0)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#%% NORMAL TRAIN
model =Sequential()
model.add(Dense(units = 120, init = 'uniform', activation = 'softsign', input_dim = 155))
model.add(Dense(units = 120, init = 'uniform', activation = 'softsign'))
model.add(Dense(units = 120, init = 'uniform', activation = 'softsign'))
model.add(Dense(units = 120, init = 'uniform', activation = 'softsign'))
model.add(Dense(units = 19, init = 'uniform', activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 100, epochs = 2000)



#%%CROSS VAL

def model():
    classifier = Sequential()
    classifier.add(Dense(units=78, init='uniform', activation='tanh', input_dim=155))
    classifier.add(Dense(units=78, init='uniform', activation='tanh'))
    classifier.add(Dense(units=19, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
classifier = KerasClassifier(build_fn=model, batch_size=10, epochs=500)
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10, n_jobs=-1)
variance = accuracies.std()
mean = accuracies.mean()

print(mean, variance)
#%% GRIDSEARCH

def model(activation):
    classifier = Sequential()
    classifier.add(Dense(units=78, init='uniform', activation=activation, input_dim=155))
    classifier.add(Dense(units=78, init='uniform', activation=activation))
    classifier.add(Dense(units=19, init='uniform', activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

classifier = KerasClassifier(build_fn=model)
parameters = {'batch_size' : [10,25],'epochs' : [500,750], 'activation' : ['relu','softsign']}
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = "accuracy",cv = 10)
grid_search = grid_search.fit(x_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
#%%pred

pred_array = np.array([[0,4.5,169,5.6,0,0,1,1]])
enc = joblib.load("encoder.pkl")
pred_array = enc.fit_transform(pred_array).toarray()
pred_array = scaler.transform(pred_array)
single_pred = model.predict(pred_array)







  





