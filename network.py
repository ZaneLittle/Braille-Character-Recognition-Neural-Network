from PIL import Image
import tensorflow as tf
import pandas as pd
from utils import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    train_dir = 'basic_braille_dataset'
    dataset = getData(train_dir)
    

    # Define params
    # batch_size = 128
    epochs = 15
    # IMG SIZE                    
    img_height = 240
    img_width = 240

    # Define model and comple
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu',
            input_shape=(img_height, img_width, 3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

