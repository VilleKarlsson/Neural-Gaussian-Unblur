from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import numpy as np
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
#Load and scale the previously generated training and test data
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_train = x_train.astype('float32') / 255.
y_train = y_train.astype('float32') / 255.

x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
x_test = x_test.astype('float32') / 255.
y_test = y_test.astype('float32') / 255.

input_dim = Input(shape=(10,50,1))


x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_dim)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(12, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(24, (3, 3), activation='relu', padding='same')(x)

encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(24, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(12, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid')(x)

autoencoder = Model(input_dim, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

x_train = np.reshape(x_train, (len(x_train), 10, 50, 1))  
y_train = np.reshape(y_train, (len(y_train), 10, 50, 1))

x_test = np.reshape(x_test, (len(x_test), 10, 50, 1))  
y_test = np.reshape(y_test, (len(y_test), 10, 50, 1))  

autoencoder.fit(y_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(y_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])



decoded_imgs = autoencoder.predict(y_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(y_test[i].reshape(10, 50))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(10, 50))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
