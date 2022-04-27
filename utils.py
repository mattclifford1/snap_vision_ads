'''
utils for evaluate.py
'''
import os
import matplotlib.pyplot as plt
from skimage import io

def get_label(filename):
    head, _ = os.path.split(filename)
    return int(os.path.basename(head))


def plot_closest(image_path, closest_paths, row, rows, count):
    # plot the image and its closest
    row_num = (row-1)*(len(closest_paths)+1)
    ax = plt.subplot(rows, len(closest_paths)+1, row_num+1)
    plt.imshow(io.imread(image_path))
    if row == True:
        ax.set_title('Input Image')
    plt.yticks([])
    plt.xticks([])
    plt.xlabel(get_label(image_path), fontsize=7)
    plt.ylabel(count)
    count = 1
    for close_path in closest_paths:
        ax = plt.subplot(rows, len(closest_paths)+1, row_num+count+1)
        plt.imshow(io.imread(close_path))
        if row == True:
            ax.set_title(str(count))
        plt.yticks([])
        plt.xticks([])
        plt.xlabel(get_label(close_path), fontsize=7)
        count += 1
