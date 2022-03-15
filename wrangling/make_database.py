import os
import glob
import pandas as pd
from argparse import ArgumentParser


def get_all_subdirs(dir):
    all_paths = glob.glob(dir + '/**', recursive=True)
    # keep only files
    dirs = []
    for path in all_paths:
        if os.path.isdir(path):
            dirs.append(path)
    return dirs

def contruct_database(dirs):
    all_images = []
    similar = []
    for dir in dirs:
        # get all image paths
        images = []
        for image in os.listdir(dir):
            images.append(os.path.join(dir, image))
        # add images with their similar images from the dir
        print(images)
        for image in images:
            print(image)
            all_images.append(image)
            # similar.append(set(images) - set(image))
        print(' ')

    df = pd.DataFrame(data={'image_path': all_images,
                            'similar_images': similar})
    print(df.head())
    # df.to_csv('./data/image_paths.csv')



if __name__ == '__main__':
    # set up command line arguments to specifiy where the dataset is
    parser = ArgumentParser()
    parser.add_argument("--dir", default='data/uob_image_set')
    ARGS = parser.parse_args()

    all_subdirs = get_all_subdirs(ARGS.dir)
    contruct_database(all_subdirs)
