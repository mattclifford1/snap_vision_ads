# %%
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import io
from skimage.color import rgb2gray, rgb2lab
from tqdm import tqdm
import os
from scipy import stats
from exploration.colour_utilities import get_dominant_colours
import cv2


def mask(image, lower_thresh, upper_thresh):
    # Create mask to only select black
    thresh = cv2.inRange(image, lower_thresh, upper_thresh)
    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # invert morp image
    mask = 255 - morph
    # apply mask to image
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def get_simple_features(filename, apply_mask=False, lower=[200, 200, 200], upper=[255, 255, 255]):
    image = io.imread(filename)
    if apply_mask == True:
        lower = np.array(lower)
        upper = np.array(upper)
        image = mask(image, lower, upper)
    mean = np.mean(image)
    mode_red = stats.mode(image[:, :, 0], axis=None)[0][0]
    mode_green = stats.mode(image[:, :, 1], axis=None)[0][0]
    mode_blue = stats.mode(image[:, :, 2], axis=None)[0][0]
    return {'mean':mean,
            'mode_red':mode_red,
            'mode_green':mode_green,
            'mode_blue':mode_blue}

def get_dominant_colours_features(filename, apply_mask=False, lower=[200, 200, 200], upper=[255, 255, 255]):
    dominant_colours = []

    for c in get_dominant_colours(filename, count=5):
        dominant_colours.append(rgb2lab(c).tolist())


        # lab_dom_col = []
        # for col in dominant_colors:
        #     lab_dom_col.append(rgb2lab(col))

    return {'dominant_colours':dominant_colours}

if __name__ == '__main__':
    print('Testing features')
    # get data loader
    import sys
    sys.path.append('..')
    sys.path.append('.')
    from data_loader.load import get_database
    data = get_database()
    # test get_features
    # features = get_features(data[0]['image_path'])
    features = get_features(data[0]['image_path'], apply_mask=True, lower=[200, 200, 200], upper=[255, 255, 255])
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

# %%
