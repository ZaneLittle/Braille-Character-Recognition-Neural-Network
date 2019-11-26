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
    train_batch_size = 104
    validate_batch_size = 26
    epochs = 20                  
    img_height = 240
    img_width = 240
    data_dir = 'braille_mixed'

    # Define data generator
    train_image_generator = ImageDataGenerator(rescale=1./255)
    train_data_gen = train_image_generator.flow_from_directory(
        directory=data_dir + "/train", 
        shuffle=True, 
        target_size=(img_height, img_width),
        batch_size=train_batch_size
    )

    valid_image_generator = ImageDataGenerator(rescale=1./255)
    valid_data_gen = train_image_generator.flow_from_directory(
        directory=data_dir + "/valid",
        shuffle=True, 
        target_size=(img_height, img_width),
        batch_size=validate_batch_size
    )

    test_image_generator = ImageDataGenerator(rescale=1./255)
    test_data_gen = train_image_generator.flow_from_directory(
        directory=data_dir + "/test",
        shuffle=True, 
        target_size=(img_height, img_width),
        batch_size=1
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
        generator=train_data_gen,
        steps_per_epoch=train_data_gen.n//train_data_gen.batch_size,
        epochs=epochs,
        validation_data=valid_data_gen,
        validation_steps=valid_data_gen.n//valid_data_gen.batch_size,
        # verbose=0, # Suppress chatty output
        callbacks=[tensorboard_callback]
    )

    print('\nTraining results: {}\n\n'.format(train_results.history))

    # Test Model
    test_data_gen.reset()
    loss, acc = model.evaluate_generator(test_data_gen, verbose=1)
    print('Evaluation results: \n\tLoss: {} \tAccuracty: {}'.format(loss, acc))
    
    print('\n\nStarting TensorBoard... Navigate to http://localhost:6006/ for metrics breakdown.\n\n')

    os.system('tensorboard --logdir logs/')
