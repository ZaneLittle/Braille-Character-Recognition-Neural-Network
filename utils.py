import glob
import shutil
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from string import ascii_uppercase
from PIL import Image
from pathlib import Path


def join_path(paths):
    joined_path = '' + paths[0]
    for path in paths[1:]:
        joined_path += '/' + path
    return joined_path


def split_data(source, dest, ratio=0.2):
    classes = os.listdir(source)
    for letter in classes:
        files = os.listdir(source + '/' + letter)
        random.shuffle(files)
        move = files[0:int(len(files)*ratio)]
        for f in move:
            thisClass = '{}/{}'.format(dest,letter)
            if not os.path.isdir(thisClass):
                os.mkdirs(thisClass)
            shutil.move('{}/{}/{}'.format(source, letter, f), thisClass)



def copy_data(sources, dest):
    print('Copying data...')
    for source in sources: # for all data sources
        for step in os.listdir(source): # for train, test, and valid
            for letter in os.listdir(join_path([source, step])):
                files = os.listdir(join_path([source, step, letter]))
                random.shuffle(files)
                for f in files:
                    thisClass = join_path([dest, step, letter])
                    if not os.path.isdir(thisClass):
                        path = Path(thisClass)
                        path.mkdir(parents=True)
                    shutil.copy(join_path([source, step, letter, f]), thisClass)

def convert_to_letters(indecies):
    ''' Converts class indecies to letter representation '''
    numeric_map = get_numeric_mapping_dict() 
    return [numeric_map[i] for i in indecies]


def create_one_hot_from_name(image_name):
    """ Create a one-hot encoded vector from image name as labels are not declared"""
    word_label = image_name.split('.')[-3] # Doesn't include For ex,(.CSV, .PNG)
    return letter_to_one_hot(word_label)


def get_alpha_mapping_dict():
    ''' Returns a dictionary of the alphabet with class labels (A=0, B=1, ...) with character as key '''
    alpha_dict = {}
    i = 0
    for c in ascii_uppercase:
        alpha_dict[c] = i
        i += 1
    return alpha_dict


def get_numeric_mapping_dict():
    ''' Returns a dictionary of the alphabet with class labels (A=0, B=1, ...) with number as key '''
    alpha_dict = {}
    i = 0
    for c in ascii_uppercase:
        alpha_dict[i] = c
        i += 1
    return alpha_dict


def letter_to_one_hot(letter):
    ''' Convert uppercase letter to one hot vector '''
    vec = np.zeros((26,))
    vec[get_alpha_mapping_dict()[letter]] = 1
    return vec


def one_hot_to_letter(vec):
    ''' Takes a one hot vector and converts it to the corresponding uppercase letter'''
    return get_numeric_mapping_dict()[np.argmax(vec)]           


def getData(dir):
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


def plot_confusion_matrix(y_true, 
                        y_pred, 
                        normalize=False,
                        title='Results Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    Plots a 3d confusion matrix
    y_true is the list true values
    y_pred is the list of actual values
    """

    classes=set(y_true)

    cm = confusion_matrix(y_true, y_pred, labels=classes)
  
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax