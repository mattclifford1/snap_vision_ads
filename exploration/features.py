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
    print('Testing features')
    # get data loader
    import sys
    sys.path.append('..')
    sys.path.append('.')
    from data_loader.load import get_database
    data = get_database()
    # test get_features
    features = get_features(data[0]['image_path'])
    print('Features are: ', features)
    for key in features.keys():
        feature_type = type(features[key])
        if feature_type in [float, int]:
            pass
        elif np.issubdtype(feature_type, np.number):
            pass
        else:
            raise Exception('Feature \'' +str(key)+'\' must be int, float or np.number not '+str(feature_type))
    print('Features will work with the simple model :)')
