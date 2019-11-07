from PIL import Image
import tensorflow as tf
import glob
import pandas as pd
from string import ascii_uppercase

def getData(dir='basic_braille_dataset'):
    """
    Returns a dataframe of the 
    dir: the directory the dataset is stored in. 
    Each image (character example) must be stored under a folder named as the respective letter
    """
    data_arr = []
    for letter in ascii_uppercase:
        tmp = []
        for filename in glob.glob(dir + '/' + letter + '/*'):
            img = Image.open(filename)
            data_arr.append([img, letter])
    return pd.DataFrame(data=data_arr, columns=['img', 'class'])

if __name__ == '__main__':
    dataset = getData()
    print(dataset) # debug
