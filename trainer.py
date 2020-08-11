#!/usr/bin/env python
# coding: utf-8

import numpy as np

print("Loading pre-processed image data ")
data=np.load('imageData.npy')
y=np.load('target.npy')
print("Done, data size: ", data.shape)

from keras.utils import np_utils
target=np_utils.to_categorical(y)

# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
print("Calculating Class weights (Imbalanced data)")
total = len(data)
mask_count = len(y[y==1])
unmask_count= total - mask_count
print(total, unmask_count, mask_count)
weight_for_0 = (1 / unmask_count)*(total)/2.0 
weight_for_1 = (1 / mask_count)*(total)/2.0

class_weights = {0: weight_for_0, 1: weight_for_1}
print("Done, class_weights: ", class_weights)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Flatten,Dropout
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import initializers

initializer = initializers.GlorotUniform()

print("Compiling ML models")
model=Sequential()

model.add(Conv2D(32,(3,3),input_shape=data.shape[1:], kernel_initializer=initializer, padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The first CNN layer followed by Relu and MaxPooling layers

model.add(Conv2D(64, (3,3), kernel_initializer=initializer, padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The second convolution layer followed by Relu and MaxPooling layers

model.add(Conv2D(128, (3,3), kernel_initializer=initializer, padding="same", strides=(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The third convolution layer followed by Relu and MaxPooling layers

model.add(Flatten())
model.add(Dropout(0.5))
#Flatten layer to stack the output convolutions from second convolution layer
model.add(Dense(1024, activation='relu', kernel_initializer=initializer))
#Dense layer of 64 neurons
model.add(Dense(2,activation='softmax', kernel_initializer=initializer))
#The Final layer with two outputs for two categories

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'], class_weight=class_weights)
print(model.summary())
print("Done")


from sklearn.model_selection import train_test_split
print("Spliting dataset ratio : 80:20")
train_data,test_data,train_target,test_target=train_test_split(data, target, stratify=y, random_state=0, test_size=0.2)

print(train_data.shape, test_data.shape)

print("Starting training")
checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history=model.fit(train_data,train_target,epochs=20,callbacks=[checkpoint],validation_split=0.2)
print("Training Done")


print("evaluate model with test_data")
print(model.evaluate(test_data,test_target))


from matplotlib import pyplot as plt

plt.plot(history.history['loss'],'r',label='training loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


plt.plot(history.history['accuracy'],'r',label='training accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.show()