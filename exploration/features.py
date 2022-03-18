# %%
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import io
from skimage.color import rgb2gray
from tqdm import tqdm
import os
from scipy import stats


def get_features(filename):
    image = io.imread(filename)
    mean = np.mean(image)
    mode_red = stats.mode(image[:, :, 0], axis=None)[0][0]
    mode_green = stats.mode(image[:, :, 1], axis=None)[0][0]
    mode_blue = stats.mode(image[:, :, 2], axis=None)[0][0]
    return {'mean':mean,
            'mode_red':mode_red,
            'mode_green':mode_green,
            'mode_blue':mode_blue}


if __name__ == '__main__':
    # get data loader
    import sys
    sys.path.append('..')
    sys.path.append('.')
    from data_loader.load import get_database
    data = get_database()
    # test get_features
    print(get_features(data[0]['image_path']))
