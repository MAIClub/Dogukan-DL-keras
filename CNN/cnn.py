
#%%
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense,Dropout

model = Sequential()
model.add(Convolution2D(32,3,3,input_shape= (128,128,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(64,3,3,activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(64,3,3,activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32,3,3,activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(output_dim=128, activation = 'relu'))
model.add(Dense(output_dim=128, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(output_dim=1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics=['accuracy'])

#%%
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')
#%%
model.fit_generator(
        training_set,
        steps_per_epoch=8000/32,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000/32)
#%%Save model
model.save('my_model.h5')
#%%LOAD MODEL
from keras.models import load_model
mdl = load_model('my_model.h5')
#%% PREDICTION
import numpy as np
from keras_preprocessing import image
predimage =  image.load_img('dataset/single_prediction/asdasd.jpg',target_size=(128,128))
predimage = image.img_to_array(predimage)
predimage = np.expand_dims(predimage,axis = 0)
result = model.predict(predimage)

if result[0][0] == 1:
        prediction = 'dog'
else:
        prediction = 'cat'
print(prediction)

#%%
