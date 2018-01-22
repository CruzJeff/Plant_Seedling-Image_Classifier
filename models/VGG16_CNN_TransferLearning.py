# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 23:25:25 2018

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 04:13:48 2018

@author: User
"""

#Importing libraries
import numpy as np
import pandas as pd
import os
from os import listdir
from keras.preprocessing import image
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer

#Path to data
TRAIN_PATH = ".//train//train"  
TEST_PATH = ".//test//test" 
    
#Loading in the data
classes = listdir(TRAIN_PATH)

def read_img(data_dir,filepath, size):
    img = image.load_img(os.path.join(data_dir, filepath), target_size=size)
    img = image.img_to_array(img)
    return img

train = []

for index, label in enumerate(classes):
    path = TRAIN_PATH + '//' + label 
    for file in listdir(path):
        train.append(['{}/{}'.format(label, file), label, index])
test = []
path = TEST_PATH
for file in listdir(path):
    test.append(file)

#Creating training and validation sets
train = pd.DataFrame(train, columns=['file', 'category', 'category_id',]) 
filepaths = train['file']

X_train = np.array([read_img(TRAIN_PATH,file,(48,48)) for file in filepaths ])
y_train = np.array(train['category'])
lb = LabelBinarizer().fit(y_train)
y_train = lb.transform(y_train)

X_train,y_train = shuffle(X_train,y_train)

X_val = X_train[3785:]
y_val = y_train[3785:]
X_train = X_train[:3785]
y_train = y_train[:3785]


X_test = [read_img(TEST_PATH,file,(48,48)) for file in test]

def f2_score(y_true, y_pred):
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(tf.round(y_pred), "int32") # implicit 0.5 threshold via tf.round
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)
    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true
    f_score = 5 * precision * recall / (4 * precision + recall)
    f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
    return tf.reduce_mean(f_score)

#Creating the CNN
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(48,48,3))


model = Sequential()
model.add(conv_base)
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Flatten())  
model.add(Dense(12))
model.add(Activation('softmax'))

conv_base.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[f2_score])


model.fit(X_train, y_train, 
          validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=True)
    
#Make Predictions
X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 3)
y_pred = model.predict(X_test)
y_pred = lb.inverse_transform(y_pred)

#Output Predictions.csv
submission = pd.DataFrame({'File':test, 'Species': y_pred.reshape((y_pred.shape[0]))})
submission.to_csv("./SubmissionTransferLearning.csv", index=False)

    
model.save('VGG16_TransferLearning_Model')