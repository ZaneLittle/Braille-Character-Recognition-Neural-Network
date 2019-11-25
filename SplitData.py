import shutil
import os
import random

source = 'braille_uncropped/train'
dest = 'braille_uncropped/test'

classes = os.listdir(source)
for letter in classes:
    files = os.listdir(source + '/' + letter)
    random.shuffle(files)
    move = files[0:int(len(files)*0.2)] # move 20%
    for f in move:
        shutil.move('{}/{}/{}'.format(source, letter, f), '{}/{}'.format(dest,letter))
