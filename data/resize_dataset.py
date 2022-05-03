'''
downscale dataset to lower resolution images for ease of computations
'''
import os
from skimage import io, transform
import numpy as np
from tqdm import tqdm


def run(data_dir, im_size=512):
    for dir in tqdm(os.listdir(data_dir), desc="Downscaling image dataset to: "+str(im_size), leave=False):
        orig_dir = os.path.join(data_dir, dir)
        new_dir = os.path.join(data_dir+str(im_size), dir)
        
        try:
            for im in os.listdir(orig_dir):
                orig_im_path = os.path.join(orig_dir, im)
                new_im_path = os.path.join(new_dir, im)
                if os.path.isfile(new_im_path):
                    continue # already saved this im previously
                else:
                    image = io.imread(orig_im_path)
                    down_im = transform.resize(image, (im_size, im_size))
                    check_write_path(new_im_path)
                    io.imsave(new_im_path, (down_im*255).astype(np.uint8))
        except:
            print('WARNING: No image found at following directory.\n',print(orig_dir))

    return data_dir+str(im_size)

def check_write_path(path):
    dir = os.path.dirname(path)
    if not os.path.isdir(dir):
        os.makedirs(dir)
