import os
import glob
import pandas as pd
from argparse import ArgumentParser


def get_all_subdirs(dir):
    dirs = glob.glob(dir + '/**/', recursive=True)
    dirs.pop(0)  # remove first one with is the base dir
    return dirs

def contruct_database(dirs):
    all_images = []
    similar = []
    for dir in dirs:
        # get all image paths
        images = []
        for image_file in os.listdir(dir):
            images.append(os.path.join(dir, image_file))
        # add images with their similar images from the dir
        for image_path in images:
            all_images.append(image_path)
            similar.append(list(set(images) - set(image_path)))
        # print(' ')
    # print(all_images)
    df = pd.DataFrame(data={'image_path': all_images,
                            'similar_images': similar})
    # print(df.head())
    df.to_csv('image_paths.csv', index=False)


if __name__ == '__main__':
    # set up command line arguments to specifiy where the dataset is
    parser = ArgumentParser()
    parser.add_argument("--dir", default='data/uob_image_set')
    ARGS = parser.parse_args()

    all_subdirs = get_all_subdirs(ARGS.dir)
    contruct_database(all_subdirs)
