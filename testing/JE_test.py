# %%
# %load_ext autoreload

# %autoreload 2


# %%
from __future__ import print_function
from skimage import io, color,filters
import numpy as np
import os
from tqdm import tqdm

import binascii
import struct
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
from sklearn.cluster import KMeans
from os import walk

# Nasty workaround for running from subdirectory
import sys
sys.path.insert(0,sys.path[0][0:sys.path[0].rfind('/')])

# %%
from exploration.colour_utilities import get_dominant_colours, choose_tint_color, coloured_square
# from .exploration import colour_utilities

set_ID = '11059585'
photo_ID = '0'

conc_ID = set_ID + '/' + set_ID + '_' + photo_ID + '.jpg'

path = '../data/uob_image_set/'

filename = os.path.join(path,conc_ID)
# filename = '/Users/jonathanerskine/Courses/Applied_Data_Science/CWK/snap_vision_ads/snap_vision_ads/data/uob_image_set/11059585/11059585_0.jpg'
filename = '/Users/jonathanerskine/Courses/Applied_Data_Science/CWK/snap_vision_ads/snap_vision_ads/data/uob_image_set/11059585/11059585_0.jpg'
image = io.imread(filename)

# %%
n = 10

# %%
#  create positive and negative examples
# path | id | number | dom_colours | 
set1 = '/Users/jonathanerskine/Courses/Applied_Data_Science/CWK/snap_vision_ads/snap_vision_ads/data/uob_image_set/11059585'
set2 = '/Users/jonathanerskine/Courses/Applied_Data_Science/CWK/snap_vision_ads/snap_vision_ads/data/uob_image_set/11233998'
set3 = '/Users/jonathanerskine/Courses/Applied_Data_Science/CWK/snap_vision_ads/snap_vision_ads/data/uob_image_set/16013076'
set4 = '/Users/jonathanerskine/Courses/Applied_Data_Science/CWK/snap_vision_ads/snap_vision_ads/data/uob_image_set/16196265'
set5 = '/Users/jonathanerskine/Courses/Applied_Data_Science/CWK/snap_vision_ads/snap_vision_ads/data/uob_image_set/16307461'
sets = [set1,set2,set3,set4,set5]
# get all filenames for each set


set_paths = []

data = {'path':[],'dom_colours_rgb':[],'dom_colours_lab':[]}

for set in sets:
    f = []
    for (dirpath, dirnames, filenames) in walk(set):
        f.extend(filenames)
        break

    # set_paths.append(f)
    for im in f:
        im_path = set+'/'+im
        data['path'].append(im_path)

        dominant_colors = get_dominant_colours(im_path, count=n)

        lab_dom_col = []
        for col in dominant_colors:
            lab_dom_col.append(color.rgb2lab(col))

        
        data['dom_colours_rgb'].append(dominant_colors)
        data['dom_colours_lab'].append(lab_dom_col)

test_image = {'path':data['path'].pop(0),
              'dom_colours_rgb':data['dom_colours_rgb'].pop(0),
              'dom_colours_lab':data['dom_colours_lab'].pop(0)}
# %%
for k in range(0,len(data['path'])):
    comp_im = {'path':data['path'][k],
              'dom_colours_rgb':data['dom_colours_rgb'][k],
              'dom_colours_lab':data['dom_colours_lab'][k]}
    A = np.zeros([n,n])

    i = 0
    j = 0

    col_data = []

    # for d in data['dom_colours_lab'][0]:
    for t in range(0,len(test_image['dom_colours_lab'])):
        row_data = []
        for c in range(0,len(comp_im['dom_colours_lab'])):
            row_data.append({'test_col':test_image['dom_colours_lab'][t],
                    'im_col':comp_im['dom_colours_lab'][c],
                    'test_col_rgb':test_image['dom_colours_rgb'][t],
                    'im_col_rgb':comp_im['dom_colours_rgb'][c],
                    'distance':color.deltaE_cie76(comp_im['dom_colours_lab'][c],
                    test_image['dom_colours_lab'][t])})

            print('comp',comp_im['dom_colours_lab'][c])
            A[t,c] = color.deltaE_cie76(comp_im['dom_colours_lab'][c],
                                        test_image['dom_colours_lab'][t])
        col_data.append(row_data)

    inds_A = scipy.optimize.linear_sum_assignment(A)

    print("Image ",str(k))
    sum_dist = 0
    for n_A in range(0,len(inds_A[0])):
        i = inds_A[0][n_A]
        j = inds_A[1][n_A]
        col = [col_data[i][j]['test_col_rgb'],col_data[i][j]['im_col_rgb']]
        
        squares = [coloured_square("#%02x%02x%02x" % tuple(int(v * 255) for v in col[0])),coloured_square("#%02x%02x%02x" % tuple(int(v * 255) for v in col[1]))]
        
        print(squares[0],squares[1],' | Distance: ',col_data[i][j]['distance'])
        sum_dist = sum_dist + col_data[i][j]['distance']
    
    print("Total Distance: ", sum_dist)

