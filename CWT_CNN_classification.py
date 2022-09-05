
import numpy as np
import os
import tensorflow as tf
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import History
from keras.utils import np_utils


def loadList(filename):
    # the filename should mention the extension 'npy'
    tempNumpyArray=np.load(filename)
    return tempNumpyArray.tolist()


CWT_dataset_path = r'C:\Users\cnzak\Desktop\data\CWT_dataset'
y_train = loadList(os.path.join(CWT_dataset_path, 'y_train_file.npy').replace("\\", "/"))
y_test = loadList(os.path.join(CWT_dataset_path, 'y_test_file.npy').replace("\\", "/"))
x_train = np.load(os.path.join(CWT_dataset_path, 'x_train_file.npy').replace("\\", "/"))
x_test = np.load(os.path.join(CWT_dataset_path, 'x_test_file.npy').replace("\\", "/"))

# --------------------------------------------------------------------------------
# Training the convolutional neural network with the CWT
# --------------------------------------------------------------------------------

history = History()

img_x = 127
img_y = 127
img_z = 9
input_shape = (img_x, img_y, img_z)

num_classes = 6
batch_size = 16
num_classes = 7
epochs = 10

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=tf.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])

train_score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))
