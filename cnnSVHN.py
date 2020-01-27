import os

import numpy as np
from scipy.io import loadmat
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

np.random.seed(1400)
import itertools
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from six.moves import urllib
import matplotlib.pyplot as plt

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


def get_kfold(x_train, y_train, k):
    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(x_train, y_train))
    return folds, x_train, y_train


def get_model(x_train):
    model = Sequential()
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

    return model


def plot_confusion_matrix(confusion_matrix, target_names, normalize=False, title='Confusion matrix'):
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = confusion_matrix.max() / 1.5 if normalize else confusion_matrix.max() / 2

    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(confusion_matrix[i, j]),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(confusion_matrix[i, j]),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def get_random_index_of_images():
    indexes_for_ran_chosen_image_each_class = dict()
    indexes_for_ran_chosen_image_each_class[1] = 9
    indexes_for_ran_chosen_image_each_class[2] = 2
    indexes_for_ran_chosen_image_each_class[3] = 3
    indexes_for_ran_chosen_image_each_class[4] = 15
    indexes_for_ran_chosen_image_each_class[5] = 5
    indexes_for_ran_chosen_image_each_class[6] = 21
    indexes_for_ran_chosen_image_each_class[7] = 14
    indexes_for_ran_chosen_image_each_class[8] = 13
    indexes_for_ran_chosen_image_each_class[9] = 1

    return indexes_for_ran_chosen_image_each_class


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

x_train_validation_data = x_train[:7326]
y_train_validation_data = y_train[:7326]

x_train = x_train[7326:]
y_train = y_train[7326:]

es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=2)
checkpoint = ModelCheckpoint('/logs/logs.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

model = get_model(x_train)
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_data=(x_train_validation_data, y_train_validation_data),
          callbacks=[es])

predicted_values = model.predict_classes(x_test)

matrix = metrics.confusion_matrix(y_test, predicted_values)
print(metrics.accuracy_score(y_test, predicted_values))
print(metrics.f1_score(y_test, predicted_values, average='micro'))
print(metrics.recall_score(y_test, predicted_values, average='micro'))
print(metrics.precision_score(y_test, predicted_values, average='micro'))

target_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
plot_confusion_matrix(matrix, target_names)
