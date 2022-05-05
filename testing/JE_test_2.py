# %%
# %load_ext autoreload

# %autoreload 2

%matplotlib inline
# %%
import csv
# from __future__ import print_function
from skimage import io, color,filters
import numpy as np
import os
from tqdm import tqdm

import binascii

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import struct
import pickle
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
from sklearn.cluster import KMeans
import torch.multiprocessing as multiprocessing
from os import walk

# Nasty workaround for running from subdirectory
import sys
sys.path.insert(0,sys.path[0][0:sys.path[0].rfind('/')])

from exploration.colour_utilities import get_dominant_colours, choose_tint_color, coloured_square

# %%
def get_file_cols(folder):
    # in windows:
    # folder = r'C:\a_folder\'
    # or folder = 'C:/a_folder/'

    image_data = {'path':[],'dominant_colours':[]}
    file_data = []

    for dirname, dirs, files in os.walk(folder):
        for filename in files:
            filename_without_extension, extension = os.path.splitext(filename)
            if extension == '.jpg':
                file_data.append(dirname+'/'+ filename)

    for file in tqdm(file_data):
        image_data['path'].append(file)
        image_data['dominant_colours'].append([color.rgb2lab(col) for col in get_dominant_colours(file)])

    return image_data
        # for filename in files:
            # filename_without_extension, extension = os.path.splitext(filename)
            # if extension == '.jpg':
                # fname = dirname+'/'+filename
                # image_data['path'].append(fname)
                # image_data['dominant_colours'].append(get_dominant_colours(fname))

# %%
image_data = get_file_cols('/Users/jonathanerskine/Courses/Applied_Data_Science/CWK/snap_vision_ads/snap_vision_ads/data/uob_image_set')

with open('/Users/jonathanerskine/Courses/Applied_Data_Science/CWK/snap_vision_ads/snap_vision_ads/testing/im_data.pickle', 'wb') as handle:
    pickle.dump(image_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%

with open('/Users/jonathanerskine/Courses/Applied_Data_Science/CWK/snap_vision_ads/snap_vision_ads/testing/im_data.pickle', 'rb') as handle:
    image_data_dict = pickle.load(handle)

# %%
def get_closest_colours(test_image,comparison_images, n = 5):
    
    result = {'label':[],'distance':[]}
    test_im = test_image['dominant_colours']
    test_label = test_image['path']

    for m in range(len(comparison_images['dominant_colours'])):
        
        
        comp_im = comparison_images['dominant_colours'][m]

        A = np.zeros([n,n])

        i = 0
        j = 0

        # for d in data['dom_colours_lab'][0]:
        for t in range(0,n):
            for c in range(0,n):
                # print('N: ',n,' | T: ',t,' | C: ',c)
                # print('-'*100)
                # print('Embedding Colour: ',e[c],'\n Test Colour: ',dom_cols[t],'\n Distance: ',color.deltaE_cie76(e[c],dom_cols[t]))
                A[t,c] = color.deltaE_cie76(comp_im[c],test_im[t])

        inds_A = scipy.optimize.linear_sum_assignment(A)

        sum_dist = 0
        for n_A in range(0,len(inds_A[0])):
            i = inds_A[0][n_A]
            j = inds_A[1][n_A]
        
            sum_dist = sum_dist + A[i][j]

        result['distance'].append(sum_dist)
        result['label'].append(comparison_images['path'][m][107:])

    nn = int(n+1)
    inds = np.argpartition(result['distance'], nn)[:nn].tolist()
    # print('INDICES: ',inds, '\n Distances: ',[result['distance'][idx] for idx in inds])
    sort_inds = np.argsort([result['distance'][idx] for idx in inds])
    # print('Sorted INDICES: ',[inds[ins] for ins in sort_inds])
    labels_out = [result['label'][idx] for idx in sort_inds] 
    return labels_out
# %%

results_labels = []
results = {'id':[],'set_id':[]}
for k in tqdm(range(len(image_data_dict['path']))):
    test_image = {'path':image_data_dict['path'][k],
                  'dominant_colours':image_data_dict['dominant_colours'][k]}

    n = len(test_image['dominant_colours'])
    results_labels.append(get_closest_colours(test_image,image_data_dict,n))

with open('/Users/jonathanerskine/Courses/Applied_Data_Science/CWK/snap_vision_ads/snap_vision_ads/testing/result_labels.pickle', 'wb') as handle:
    pickle.dump(results_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%

with open('/Users/jonathanerskine/Courses/Applied_Data_Science/CWK/snap_vision_ads/snap_vision_ads/testing/result_labels.pickle', 'rb') as handle:
    results_labels_pickle = pickle.load(handle)

# %%
closest = 0
top_5 = 0
for labels in tqdm(results_labels_pickle):
    if labels[0] in labels[1]:
        closest += 1
    if labels[0] in labels[1:]:
        top_5 += 1

# %% Evaluation
for i in range(len(image_data_dict['path'])):
    n = len(results_labels_pickle[0])

    fig = plt.figure()
    ax = fig.add_subplot(1, n, 1)
    imgplot = plt.imshow(lum_img)
    ax.set_title('Before')
    plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(lum_img)
    imgplot.set_clim(0.0, 0.7)
    ax.set_title('After')
    plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    for 
    img = mpimg.imread('../../doc/_static/stinkbug.png')
    print(img)
    imgplot = plt.imshow(img)
