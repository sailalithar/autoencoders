# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 23:39:59 2017

@author: saila

"""
#import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import History 
history = History()
#from autoencoder import Autoencoder

# this is the size of our encoded representations
encoding_dim = 36  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)

input_img = Input(shape=(784,))
decoder_input = Input(shape=(encoding_dim,), name='DecoderIn')
decoder = decoder_input

encoded = Dense(100, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(100, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)


autoencoder = Model(input_img, decoded)

# this is vanilla RNN
#autoencoder = Model(input_img, decoded)
#this is RNN with sparsity

encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input_1 = Input(shape=(encoding_dim,))
encoded_input_2 = Input(shape=(64,))
encoded_input_3 = Input(shape=(100,))

# retrieve the last layer of the autoencoder model
decoder_layer_1 = autoencoder.layers[-3]
decoder_layer_2 = autoencoder.layers[-2]
decoder_layer_3 = autoencoder.layers[-1]

# create the decoder model
decoder_1 = Model(encoded_input_1,decoder_layer_1(encoded_input_1))
decoder_2 = Model(encoded_input_2,decoder_layer_2(encoded_input_2))
decoder_3 = Model(encoded_input_3,decoder_layer_3(encoded_input_3))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[history])
# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs1 = decoder_1.predict(encoded_imgs)
decoded_imgs2 = decoder_2.predict(decoded_imgs1)
decoded_imgs3 = decoder_3.predict(decoded_imgs2)
#  "Accuracy"
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


n = 10  # how many digits we will display
plt.figure(figsize=(50, 4))
for i in range(n):
    # display original
    ax = plt.subplot(5, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display encoded img
    ax = plt.subplot(5, n, i + 1 + n)
    plt.imshow(encoded_imgs[i].reshape(6, 6))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
        # display decoded1 img
    ax = plt.subplot(5, n, i + 1 + n + n) 
    plt.imshow(decoded_imgs1[i].reshape(8, 8))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
            # display decoded2 img
    ax = plt.subplot(5, n, i + 1 + n + n + n) 
    plt.imshow(decoded_imgs2[i].reshape(10, 10))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(5, n, i + 1 + n + n + n +n)
    plt.imshow(decoded_imgs3[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


