import glob
import numpy as np
import pandas as pd
from string import ascii_uppercase
from PIL import Image

def create_one_hot_from_name(image_name):
    """ Create a one-hot encoded vector from image name as labels are not declared"""
    word_label = image_name.split('.')[-3] # Doesn't include For ex,(.CSV, .PNG)
    if word_label == 'A':
        return np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif word_label == 'B':
        return np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])                   
    elif word_label == 'C':
        return np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif word_label == 'D':
        return np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif word_label == 'E':
        return np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif word_label == 'F':
        return np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif word_label == 'G':
        return np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif word_label == 'H':
        return np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif word_label == 'I':
        return np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif word_label == 'J':
        return np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif word_label == 'K':
        return np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])  
    elif word_label == 'L':
        return np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])                       
    elif word_label == 'M':
        return np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])                       
    elif word_label == 'N':
        return np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])                      
    elif word_label == 'O':
        return np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]) 
    elif word_label == 'P':
        return np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])                       
    elif word_label == 'Q':
        return np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])                       
    elif word_label == 'R':
        return np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])                      
    elif word_label == 'S':
        return np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]) 
    elif word_label == 'T':
        return np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])                   
    elif word_label == 'U':
        return np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
    elif word_label == 'V':
        return np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])  
    elif word_label == 'W':
        return np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])                       
    elif word_label == 'X':
        return np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])                       
    elif word_label == 'Y':
        return np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])                      
    else: # It's Z
        return np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]) 

                 
def create_name_from_one_hot(one_hot):
    """ Create a label from one_hot"""
   
    if one_hot == np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]):
        return 'A'
    elif one_hot == np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]):
        return 'B'                    
    elif one_hot == np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]):
        return 'C'
    elif one_hot == np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]):
        return 'D'   
    elif one_hot == np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]):
        return 'E'                        
    elif one_hot == np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]):
        return 'F'                        
    elif one_hot == np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]):
        return 'G'                     
    elif one_hot == np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]):
        return 'H'
    elif one_hot == np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]):
        return 'I'                
    elif one_hot == np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]):
        return 'J'
    elif one_hot == np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]):
        return 'K'   
    elif one_hot == np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]):
        return 'L'                        
    elif one_hot == np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]):
        return 'M'                      
    elif one_hot == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]):
        return 'N'                       
    elif one_hot == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]):
        return 'O'  
    elif one_hot == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]):
        return 'P'                        
    elif one_hot == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]):
        return 'Q'                        
    elif one_hot == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]):
        return 'R'                     
    elif one_hot == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]):
        return 'S'
    elif one_hot == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]):
        return 'T'                    
    elif one_hot == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]):
        return 'U'
    elif one_hot == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]):
        return 'V'   
    elif one_hot == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]):
        return 'W'                        
    elif one_hot == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]):
        return 'X'                        
    elif one_hot == 'Y':
        return 'Y'                       
    else: # It's Z
        return 'Z'                      


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
