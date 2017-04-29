from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
import keras.optimizers as optimizers
from keras.preprocessing import sequence
from keras.datasets import imdb
import numpy as np

input_dim = 80
timesteps = 20
latent_dim = 32
nb_samples = 1250

inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim)(inputs)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)

# generate dummy training data
##########################################################################################################

max_features = 20000
maxlen = input_dim  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), len(y_train), ': train sequences from IMBDB')
print(len(x_test), len(y_test), ': test sequences from IMDB')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('X_train shape:', x_train.shape)
print('X_test shape:', x_test.shape)

new_num_samples_train = int(x_train.shape[0]/timesteps)
new_num_samples_test = int(x_test.shape[0]/timesteps)
x_train = x_train.reshape(new_num_samples_train, timesteps, x_train.shape[1])
x_test = x_test.reshape(new_num_samples_test, timesteps, x_test.shape[1])
print('X_train shape after reshaping:', x_train.shape)
print('X_test shape after reshaping:', x_test.shape)

sequence_autoencoder.summary()

# opt = optimizers.RMSprop()
opt = optimizers.SGD(nesterov=True)
# opt = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
sequence_autoencoder.compile(optimizer=opt, loss='kullback_leibler_divergence')

# we don't need the sequence 'Y' values ..
sequence_autoencoder.fit(x_train, x_train,
                         epochs=500,
                         batch_size=batch_size,
                         shuffle=True,
                         validation_data=(x_test, x_test))

ae_features = sequence_autoencoder.layers[-1]


print(ae_features)