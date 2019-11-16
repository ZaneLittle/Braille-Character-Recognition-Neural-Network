import tensorflow as tf
import pandas as pd

import os

from utils import *
from datetime import datetime
from packaging import version
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    # Check Tensorflow version of current machine
    print("Using TensorFlow version: ", tf.__version__)
    assert version.parse(tf.__version__).release[0] >= 2, \
        "TensorFlow 2.0 or above is required."
    
    # Set logging directory
    logdir = ".\\logs\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    # Define params
    # batch_size = 128
    epochs = 5                  
    img_height = 240
    img_width = 240
    data_dir = 'basic_braille_dataset'


    # Define data generator
    # dataset = getData(train_dir)
    train_image_generator = ImageDataGenerator(rescale=1./255)
    train_data_gen = train_image_generator.flow_from_directory(
        directory=data_dir + "/train", 
        shuffle=True, 
        target_size=(img_height, img_width)
    )

    valid_image_generator = ImageDataGenerator(rescale=1./255)
    valid_data_gen = train_image_generator.flow_from_directory(
        directory=data_dir + "/valid", 
        shuffle=True, 
        target_size=(img_height, img_width)
    )

    test_image_generator = ImageDataGenerator(rescale=1./255)
    test_data_gen = train_image_generator.flow_from_directory(
        directory=data_dir + "/test", 
        shuffle=True, 
        target_size=(img_height, img_width)
    )


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
    train_results = model.fit_generator(
        train_data_gen,
        epochs=epochs,
        validation_data=valid_data_gen,
        # verbose=0, # Suppress chatty output
        callbacks=[tensorboard_callback]
    )

    print('\nTraining results: {}\n\n'.format(train_results.history))

    # Test Model
    test_data_gen.reset()
    pred = model.predict_generator(test_data_gen, verbose=1)
    print('Predicted output is: {}'.format(np.argmax(pred, axis=1)))
    
    print('\n\nStarting TensorBoard... Navigate to http://localhost:6006/ for metrics breakdown.\n\n')

    os.system('tensorboard --logdir logs/')
