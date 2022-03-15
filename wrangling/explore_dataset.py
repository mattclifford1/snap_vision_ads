import os
import glob
from argparse import ArgumentParser


def get_all_subdirs(dir):
    all_paths = glob.glob(dir + '/**', recursive=True)
    # keep only files
    dirs = []
    for path in all_paths:
        if os.path.isdir(path):
            dirs.append(path)
    return dirs

def get_all_files(dir):
    # dir is the base of the dataset eg. 'data/uob_image_set'
    all_paths = glob.glob(dir + '/**', recursive=True)
    # keep only files
    files = []
    for path in all_paths:
        if os.path.isfile(path):
            files.append(path)
    return files

def print_average_dir_contents(dirs):
    contents = []
    for dir in dirs:
        contents.append(len(os.listdir(dir)))
    print('Mean images in dir: '+str(sum(contents)/len(contents)))

def get_all_file_type(files):
    # explore all the different file extensions
    counter_dict = {}
    for file in files:
        text, extension = os.path.splitext(file)
        # add extension type to the counter
        if extension in counter_dict.keys():
            counter_dict[extension] += 1
        else:  # initialise new extension
            counter_dict[extension] = 1
    return counter_dict


if __name__ == '__main__':
    # set up command line arguments to specifiy where the dataset is
    parser = ArgumentParser()
    parser.add_argument("--dir", default='data/uob_image_set')
    ARGS = parser.parse_args()

    all_subdirs = get_all_subdirs(ARGS.dir)
    print('Num of dirs: '+ str(len(all_subdirs)))
    print_average_dir_contents(all_subdirs)
    all_files = get_all_files(ARGS.dir)
    print('Num of files: '+ str(len(all_files)))
    exts = get_all_file_type(all_files)
    print('Different files extensions:')
    for key in exts.keys():
        print('   '+key+': '+str(exts[key]))
