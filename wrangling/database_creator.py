import os
import glob
import pandas as pd
from argparse import ArgumentParser


def get_all_subdirs(dir):
    # check if data is downloaded
    if not os.path.isdir(dir):
        # raise FileNotFoundError('No directory: '+dir+'. Please download and unzip the dataset')
        import sys
        sys.path.append('..')
        sys.path.append('.')
        import data.download
    # find all the dirs
    dirs = glob.glob(os.path.join(dir, '**/'), recursive=True)
    dirs.pop(0)  # remove first one with is the base dir
    return dirs

def contruct_database(dir='data/uob_image_set'):
    all_subdirs = get_all_subdirs(dir)
    all_images = []
    similar = []
    labels = []
    for dir in all_subdirs:
        # get all image paths
        images = []
        for image_file in os.listdir(dir):
            images.append(os.path.join(dir, image_file))
            head, label = os.path.split(dir)
            if label == '':   # dir ended in '/'
                _, label = os.path.split(head)
            labels.append(label)
        # add images with their similar images from the dir
        for image_path in images:
            all_images.append(image_path)
            similar.append(list(set(images) - set([image_path])))
        # print(' ')
    # print(all_images)
    df = pd.DataFrame(data={'image_path': all_images,
                            'similar_images': similar,
                            'label': labels})
    # print(df.head())
    folder = 'wrangling'
    save_file = 'image_paths.csv'
    if os.path.isdir(folder):
        save_file = os.path.join(folder, save_file)
    df.to_csv(save_file, index=False)


if __name__ == '__main__':
    # set up command line arguments to specifiy where the dataset is
    parser = ArgumentParser()
    parser.add_argument("--dir", default='data/uob_image_set')
    ARGS = parser.parse_args()
    contruct_database(ARGS.dir)
