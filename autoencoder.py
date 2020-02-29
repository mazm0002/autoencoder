from keras.layers import Dense, Input, Conv2D, LSTM, MaxPool2D, UpSampling2D
from sklearn.model_selection import train_test_split
from keras.models import Model
import pandas as pd
import numpy as np


# Pre-processing data
train_data=pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
test_data=pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')

# read dataset
train_x = train_data[list(train_data.columns)[1:]].values
train_y = train_data['label'].values

# normalize and reshape the predictors
train_x = train_x / 255

# create train and validation datasets
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)

# create test dataset
test_x=test_data[list(test_data.columns)[1:]].values
test_y=test_data['label'].values

# reshape the inputs
train_x = train_x.reshape(-1, 784)
val_x = val_x.reshape(-1, 784)

## Creating autoencoder model

# Creating the input layer
input_layer=Input(shape=(784,))

# Creating the encode layers
encode1=Dense(1500,activation='relu')(input_layer)
encode2=Dense(1000,activation='relu')(encode1)
encode3=Dense(500,activation='relu')(encode2)

# Latent View
latent_view=Dense(10,activation='sigmoid')(encode3)

# Creating the decode layers
decode1=Dense(500,activation='relu')(latent_view)
decode2=Dense(1000,activation='relu')(decode1)
decode3=Dense(1500,activation='relu')(decode2)

# Output layer
output_layer=Dense(784)(decode3)

autoencode=Model(input=input_layer,output=output_layer)

autoencode.compile(optimizer='adam',loss='mse',metrics=['accuracy'])

autoencode.fit(train_x,train_x,epochs=20,batch_size=2048,validation_data=(val_x,val_x))