# %%
# %load_ext autoreload

# %autoreload 2


# %%
import csv
# from __future__ import print_function
from skimage import io, color,filters
import numpy as np
import os
from tqdm import tqdm

import binascii
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
# from .exploration import colour_utilities
# %%
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
# for k in range(0,len(data['path'])):
#     comp_im = {'path':data['path'][k],
#               'dom_colours_rgb':data['dom_colours_rgb'][k],
#               'dom_colours_lab':data['dom_colours_lab'][k]}
#     A = np.zeros([n,n])

#     i = 0
#     j = 0

#     col_data = []

#     # for d in data['dom_colours_lab'][0]:
#     for t in range(0,len(test_image['dom_colours_lab'])):
#         row_data = []
#         for c in range(0,len(comp_im['dom_colours_lab'])):
#             row_data.append({'test_col':test_image['dom_colours_lab'][t],
#                     'im_col':comp_im['dom_colours_lab'][c],
#                     'test_col_rgb':test_image['dom_colours_rgb'][t],
#                     'im_col_rgb':comp_im['dom_colours_rgb'][c],
#                     'distance':color.deltaE_cie76(comp_im['dom_colours_lab'][c],
#                     test_image['dom_colours_lab'][t])})

#             print('comp',comp_im['dom_colours_lab'][c])
#             A[t,c] = color.deltaE_cie76(comp_im['dom_colours_lab'][c],
#                                         test_image['dom_colours_lab'][t])
#         col_data.append(row_data)

#     inds_A = scipy.optimize.linear_sum_assignment(A)

#     print("Image ",str(k))
#     sum_dist = 0
#     for n_A in range(0,len(inds_A[0])):
#         i = inds_A[0][n_A]
#         j = inds_A[1][n_A]
#         col = [col_data[i][j]['test_col_rgb'],col_data[i][j]['im_col_rgb']]
        
#         squares = [coloured_square("#%02x%02x%02x" % tuple(int(v * 255) for v in col[0])),coloured_square("#%02x%02x%02x" % tuple(int(v * 255) for v in col[1]))]
        
#         print(squares[0],squares[1],' | Distance: ',col_data[i][j]['distance'])
#         sum_dist = sum_dist + col_data[i][j]['distance']
    
#     print("Total Distance: ", sum_dist)

# abspath to a folder as a string
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
        result['label'].append(comparison_images['path'][m][107:115])

    nn = int(n+1)
    inds = np.argpartition(result['distance'], nn)[:nn].tolist()
    # print('INDICES: ',inds, '\n Distances: ',[result['distance'][idx] for idx in inds])
    sort_inds = np.argsort([result['distance'][idx] for idx in inds])
    # print('Sorted INDICES: ',[inds[ins] for ins in sort_inds])
    labels_out = [result['label'][idx] for idx in sort_inds] 
    return labels_out
# %%
#     scores = []
#     for i in tqdm(range(len(self.labels)), desc="Eval Sequential compute", leave=False):
#         score = self.get_score(i)
#         print(score)
#         scores.append(score)

    # def run_parellel(self):
    #     pool_obj = multiprocessing.Pool()
    #     scores = list(tqdm(pool_obj.imap(self.get_score, range(len(self.labels))), total=len(self.labels), desc="Eval Pararrel compute", leave=False))
    #     return scores

results_labels = []
for k in tqdm(range(len(image_data_dict['path']))):
    test_image = {'path':image_data_dict['path'][k],
                  'dominant_colours':image_data_dict['dominant_colours'][k]}

    n = len(test_image['dominant_colours'])
    results_labels.append(get_closest_colours(test_image,image_data_dict,n))

# %%

def get_labels(i):

    with open('/Users/jonathanerskine/Courses/Applied_Data_Science/CWK/snap_vision_ads/snap_vision_ads/testing/im_data.pickle', 'rb') as handle:
        image_data_var = pickle.load(handle)

    test_image = {'path':image_data_var['path'][i],
                  'dominant_colours':image_data_var['dominant_colours'][i]}

    n = len(test_image['dominant_colours'])
    labels = get_closest_colours(test_image,image_data_dict,n)
    
    return labels
# %%

pool_obj = multiprocessing.Pool()
results_labels = list(tqdm(pool_obj.map(get_labels, range(len(image_data_dict['path']))),
                                          total=len(image_data_dict['path'])))


# # %%
# def get_closest_colours(all_images_dom_cols, labels, dom_cols, num_neighbours = 5):
#     dom_cols = dom_cols[0]
#     result = {'label':[],'distance':[]}

#     for m in range(len(all_images)):
        
#         e = all_images[m][0]
#         n = len(dom_cols)
#         A = np.zeros([n,n])

#         i = 0
#         j = 0

#         # for d in data['dom_colours_lab'][0]:
#         for t in range(0,n):
#             for c in range(0,n):
#                 # print('N: ',n,' | T: ',t,' | C: ',c)
#                 # print('-'*100)
#                 # print('Embedding Colour: ',e[c],'\n Test Colour: ',dom_cols[t],'\n Distance: ',color.deltaE_cie76(e[c],dom_cols[t]))
#                 A[t,c] = color.deltaE_cie76(e[c],dom_cols[t])

#         inds_A = scipy.optimize.linear_sum_assignment(A)

#         sum_dist = 0
#         for n_A in range(0,len(inds_A[0])):
#             i = inds_A[0][n_A]
#             j = inds_A[1][n_A]
        
#             sum_dist = sum_dist + A[i][j]

#         result['distance'].append(sum_dist)
#         result['label'].append(labels[m])

#     nn = int(num_neighbours+1)
#     inds = np.argpartition(result['distance'], nn)[:nn].tolist()
#     # print('INDICES: ',inds, '\n Distances: ',[result['distance'][idx] for idx in inds])
    

#     sort_inds = np.argsort([result['distance'][idx] for idx in inds])
#     # print('Sorted INDICES: ',[inds[ins] for ins in sort_inds])
#     labels_out = [result['label'][idx] for idx in sort_inds] 
# %%
