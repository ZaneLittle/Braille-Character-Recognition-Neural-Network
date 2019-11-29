import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

import os

from utils import *
from consts import Consts

from datetime import datetime
from packaging import version
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing import image

consts = Consts()

def define_datagen():
    ''' Defines and returns the train, valid and test ImageDataGenerators '''
    train_image_generator = image.ImageDataGenerator(rescale=1./255)
    train_data_gen = train_image_generator.flow_from_directory(
        directory=consts.DATA_DIR + "/train", 
        shuffle=True, 
        target_size=consts.IMAGE_SHAPE,
        batch_size=consts.TRAIN_BATCH_SIZE,
        class_mode='categorical'
    )

    valid_image_generator = image.ImageDataGenerator(rescale=1./255)
    valid_data_gen = valid_image_generator.flow_from_directory(
        directory=consts.DATA_DIR + "/valid",
        shuffle=True, 
        target_size=consts.IMAGE_SHAPE,
        batch_size=consts.VALIDATE_BATCH_SIZE,
        class_mode='categorical'
    )

    test_image_generator = image.ImageDataGenerator(rescale=1./255)
    test_data_gen = test_image_generator.flow_from_directory(
        directory=consts.DATA_DIR + "/test",
        shuffle=True, 
        target_size=consts.IMAGE_SHAPE,
        batch_size=1,
        class_mode='categorical'
    )

    return train_data_gen, valid_data_gen, test_data_gen


def define_model():
    ''' Defines, compiles and returns model '''
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu',
            input_shape=(consts.IMAGE_SHAPE[0], consts.IMAGE_SHAPE[1], 3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu',),
        Dense(26, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])
    
    return model


def train(model):
    train_results = model.fit_generator(
        generator=train_data_gen,
        steps_per_epoch=train_data_gen.n//train_data_gen.batch_size,
        epochs=consts.EPOCHS,
        validation_data=valid_data_gen,
        validation_steps=valid_data_gen.n//valid_data_gen.batch_size,
        # verbose=0, # Suppress chatty output
        callbacks=[tensorboard_callback]
    )

    print('\nTraining results: {}\n\n'.format(train_results.history))
    return model


def setup_data():
    if consts.MIXED:
        copy_data(consts.ALL_DATA, consts.DATA_DIR)


def cleanup_data():
    if consts.MIXED:
        shutil.rmtree(consts.DATA_DIR)


def predict_single_image(model, file):
    ''' Takes relative path of a single file and model and outputs prediction '''
    img = image.load_img(file, target_size=consts.IMAGE_SHAPE)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    pred_class = vec_to_letter(pred)

    print('Predicted class: {}'.format(pred_class))



def predict_dir(model, directory):
    ''' 
    Predicts all images in a directory with given model
    Directory must be regular tree in the same format as was used for testing
    Outputs prediction vs true, accuracy and confusion matrix
    '''
    pred_image_generator = image.ImageDataGenerator(rescale=1./255)
    pred_data_gen = pred_image_generator.flow_from_directory(
        directory=directory,
        shuffle=False, 
        target_size=consts.IMAGE_SHAPE,
        batch_size=1,
        class_mode=None
    )

    print('Thinking...\n')
    pred = model.predict_generator(generator=pred_data_gen, steps=pred_data_gen.n // pred_data_gen.batch_size)

    pred_class_indicies = np.argmax(pred, axis=1)
    true_class_indecies = pred_data_gen.labels
    
    pred_labels = convert_to_letters(pred_class_indicies) 
    true_labels = convert_to_letters(true_class_indecies)

    print('Pred labels: {}'.format(pred_labels))
    print('True labels: {}'.format(true_labels))

    acc = 0
    for true, pred in zip(true_labels, pred_labels):
        if true == pred:
            acc += 1
    acc = (acc / len(true_labels) * 100)
    print('Accuracy: {}\n'.format(acc))

    plot_confusion_matrix(y_true=true_labels, y_pred=pred_labels)
    plt.show()
    


def predict_loop(model):
    ''' 
    Loop and get user input filename and predict with given model
    Terminates when user enters nothing
    '''
    print("Entering prediction phase. Input nothing to exit.")
    mapping_dict = get_numeric_mapping_dict()
    while True:
        print('\nEnter the directory tree or file name (relative path) of an image or images to predict.')
        user_input = input('What would you like to predict? (enter nothing to exit):')
        if not user_input:
            break
        else:
            # Define image and pre-process:
            try:
                if os.path.isdir(user_input):
                    predict_dir(model=model, directory=user_input)
                elif os.path.isfile(user_input):
                    predict_single_image(model=model, file=user_input)
                else:
                    print('Error reading file: {}'.format(user_input))

            except ValueError as e:
                print('Invalid file. Please try again')
                print('Value error: {}'.format(e))
            except IOError as e:
                print('Problem reading the file "{}"'.format(user_input))
                print('I/O error({}): {}'.format(e.errno, e.strerror))



if __name__ == '__main__':
    # Check Tensorflow version of current machine
    print("Using TensorFlow version: ", tf.__version__)
    assert version.parse(tf.__version__).release[0] >= 2, \
        "TensorFlow 2.0 or above is required."
    
    # Set logging directory
    logdir = ".\\logs\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    setup_data()

    train_data_gen, valid_data_gen, test_data_gen = define_datagen()

    model = define_model()

    model = train(model)

    # Test Model
    test_data_gen.reset()
    loss, acc = model.evaluate_generator(test_data_gen, verbose=1)
    print('Evaluation results: \n\tLoss: {} \tAccuracty: {}'.format(loss, acc))
    
    cleanup_data()
    
    predict_loop(model)

    print('\n\nStarting TensorBoard... Navigate to http://localhost:6006/ for metrics breakdown.\n\n')

    os.system('tensorboard --logdir logs/')

