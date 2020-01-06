import numpy as np
from scipy.io import loadmat
import os

np.random.seed(1400)
import keras
from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPool2D, Dropout
from keras.callbacks import EarlyStopping
from six.moves import urllib
from keras.optimizers import SGD

URL_TRAIN_PATH = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
URL_TEST_PATH = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'

DOWNLOADED_FILENAME_TRAIN = 'housenumbers_training.mat'
DOWNLOADED_FILENAME_TEST = 'housenumbers_test.mat'

HEIGHT = 32
WIDTH = 32
CHANNELS = 3  # since there are rgb images
N_INPUTS = HEIGHT * WIDTH
N_OUTPUTS = 11


def download_data():
    if not os.path.exists(DOWNLOADED_FILENAME_TRAIN):
        filename, _ = urllib.request.urlretrieve(URL_TRAIN_PATH, DOWNLOADED_FILENAME_TRAIN)

    print('Found and verified file from this path: ', URL_TRAIN_PATH)
    print('Download file: ', DOWNLOADED_FILENAME_TRAIN)

    if not os.path.exists(DOWNLOADED_FILENAME_TEST):
        filename, _ = urllib.request.urlretrieve(URL_TEST_PATH, DOWNLOADED_FILENAME_TEST)

    print('Found and verified file from this path: ', URL_TEST_PATH)
    print('Download file: ', DOWNLOADED_FILENAME_TEST)


download_data()
# squeeze_me= True -> Unit 1x1 matrix dimensions are squeezed to be scalars
train_data_mat = loadmat(DOWNLOADED_FILENAME_TRAIN, squeeze_me=True)
test_data_mat = loadmat(DOWNLOADED_FILENAME_TEST, squeeze_me=True)

x_train = train_data_mat['X']
y_train = train_data_mat['y']
x_test = test_data_mat['X']
y_test = test_data_mat['y']

# num_images, height, width, num_channels
x_train = np.transpose(x_train, (3, 0, 1, 2))
x_test = np.transpose(x_test, (3, 0, 1, 2))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model = Sequential()
inputs = Input(shape=(32, 32, 3))
model.add(Conv2D(9, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
model.add(MaxPool2D(pool_size=(3, 3)))
model.add(Conv2D(36, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(3, 3)))
model.add(Conv2D(49, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(11, activation="softmax"))


model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.fit(x_train, y_train,  epochs=10, validation_data=(x_test, y_test))
