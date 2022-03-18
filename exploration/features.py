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
    mode = stats.mode(image)

    im_stats = {'mean':mean,
                'mode':mode,
                }

    return im_stats
# %%

if __name__ == '__main__':
    # get data loader
    import sys
    sys.path.append('..')
    from data_loader.load import get_database
    data = get_database()
    # test get_features
    print(get_features(data[0]['image']))
