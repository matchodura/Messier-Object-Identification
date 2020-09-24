from keras import Sequential
from keras import backend as K
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dropout, Dense, ZeroPadding2D, GlobalAveragePooling2D


def model_setup():
    ###############################
          ##Zmiana modelu##
    ###############################

    shape = 294  # tutaj zmieniamy wymiary
    channels = 3  # a tutaj kanały

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(shape, shape, channels)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(shape, shape, channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(shape, shape, channels)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(shape, shape, channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
   

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=(shape, shape, channels)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=(shape, shape, channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
   

    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', input_shape=(shape, shape, channels)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', input_shape=(shape, shape, channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
  

    model.add(Flatten())
    model.add(Dense(440, activation='relu'))
    model.add(Dropout(0.3))  # dropout 1
    model.add(Dense(220, activation='relu'))
    model.add(Dense(110, activation='softmax'))
    model.add(Dropout(0.3))  # d
    model.add(Dense(110, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam',
                  metrics=[
                      'accuracy',
                      'mse',
                      'AUC']
                  )
    

    return model

