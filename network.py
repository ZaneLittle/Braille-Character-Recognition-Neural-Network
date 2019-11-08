from PIL import Image
import tensorflow as tf
import glob
import pandas as pd
from string import ascii_uppercase
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def getData(dir='basic_braille_dataset'):
    """
    Returns a dataframe of the 
    dir: the directory the dataset is stored in. 
    Each image (character example) must be stored under a folder named as the respective letter
    """
    data_arr = []
    for letter in ascii_uppercase:
        for filename in glob.glob(dir + '/' + letter + '/*'):
            img = Image.open(filename)
            # img.resize(240, 240)
            # img.convert('LA') # Convert to greyscale
            data_arr.append([img, letter])
    return pd.DataFrame(data=data_arr, columns=['img', 'class'])


if __name__ == '__main__':
    dataset = getData()

    # Define params
    # batch_size = 128
    epochs = 15
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
