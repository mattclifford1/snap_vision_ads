# %%
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import io
from skimage.color import rgb2gray
from tqdm import tqdm
import os

def get_features(filename):

    # DELETE AFTER TEST
    set_ID = '11059585'
    photo_ID = '0'

    conc_ID = set_ID + '/' + set_ID + '_' + photo_ID + '.jpg'

    path = 'data/uob_image_set/'

    filepath = os.path.join(path,conc_ID)
    #DELETE AFTER TEST ^^


    image = io.imread(filepath)


    io.imshow(image)
    io.show()
    mean = 0
    mode = 0

    return mean,mode
# %%
