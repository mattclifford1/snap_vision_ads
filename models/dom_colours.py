'''
Simple model matching images with similar dominant colours
'''
import os
import pandas as pd
import ast
import sys
sys.path.append('.')
sys.path.append('..')
from evaluation import nearest_points

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

# %%
from exploration.colour_utilities import get_dominant_colours, choose_tint_color, coloured_square


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

        dominant_colors = get_dominant_colours(im_path, count=5)

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
    A = np.zeros([5,5])

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

            A[t,c] = color.deltaE_cie76(comp_im['dom_colours_lab'][c],
                                        test_image['dom_colours_lab'][t])
        col_data.append(row_data)
        # create matrix of similarities
        # test im | image col 1, 2, 3, 4, 5  |
        #   1     |   1-1, 1-2, 1-3, 1-4, 1-5
        #   2     |   2-1, 2-2, 2-3, 2-4, 2-5
        #   3     |   3-1, 3-2, 3-3, 3-4, 3-5
        #   4     |   4-1, 4-2, 4-3, 4-4, 4-5
        #   5     |   5-1, 5-2, 5-3, 5-4, 5-5
    # %%
    inds_A = scipy.optimize.linear_sum_assignment(A)

    # %%

    # for i in inds_A[0]:
    #     for j in inds_A[1]:
    #         col = [col_data[i][j]['test_col_rgb'],col_data[i][j]['im_col_rgb']]
    #         squares = coloured_square("#%02x%02x%02x" % tuple(int(v * 255) for v in col[0]))
    #         print(squares,' | Distance: ',col_data[i][j]['distance'])
    print("Image ",str(k))
    sum_dist = 0
    for n in range(0,len(inds_A[0])):
        i = inds_A[0][n]
        j = inds_A[1][n]
        col = [col_data[i][j]['test_col_rgb'],col_data[i][j]['im_col_rgb']]
        
        squares = [coloured_square("#%02x%02x%02x" % tuple(int(v * 255) for v in col[0])),coloured_square("#%02x%02x%02x" % tuple(int(v * 255) for v in col[1]))]
        
        print(squares[0],squares[1],' | Distance: ',col_data[i][j]['distance'])
        sum_dist = sum_dist + col_data[i][j]['distance']
    
    print("Total Distance: ", sum_dist)


# def check_features(features_csv='exploration/database.csv'):
#     # check features file exists (create if not)
#     features_csv = 'exploration/database.csv'
#     if not os.path.isfile(features_csv):
#         from exploration import save_features
#         save_features.save_simple_features(features_csv=features_csv)

def run(features_csv='exploration/database.csv'):

    # for each image in database
    # Check dominant colours against all other images 
    #  for first image i, check i+1:end
    #  for second image j, check j+1:end
    # for all images, if distance <200, then add to similar images
    #  using embeddings
    #  then run nearest_points.eval(embeddings,labels)
    #  maybe idk I aint a software engineer


    # check_features(features_csv)
    # # load features
    # df = pd.read_csv(features_csv)
    # embeddings = []
    # # get all features from dataframe
    # for row in range(df.shape[0]):
    #     # convert dict in str format to list of its values
    #     dict_row = ast.literal_eval(df['features'][row])
    #     embeddings.append(list(dict_row.values()))
    # labels = list(df['label'])

    # # eval features
    # evaluation = nearest_points.eval(embeddings, labels)  # will print out accuracy


if __name__ == '__main__':
    run()