from PIL import Image
import tensorflow as tf
import pandas as pd
from utils import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    # Define params
    # batch_size = 128
    epochs = 15                  
    img_height = 240
    img_width = 240
    train_dir = 'basic_braille_dataset'

    # Retrieve data
    # dataset = getData(train_dir)
    train_image_generator = ImageDataGenerator(rescale=1./255)
    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir, shuffle=True, target_size=(img_height, img_width))



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
        Dense(26, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    # Train model 
    results = model.fit_generator(
        train_data_gen,
        epochs=epochs,
        validation_data=train_data_gen
    )

