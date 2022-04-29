import math
from skimage import io
from skimage import transform
import numpy as np
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
plt.close()



def image():
    img_path = "C:/Users/dec2g/GitHub/snap_vision_ads/dan_stuff/test_image.jpg"
    img = io.imread(img_path)
    #io.imshow(img)
    #plt.show()
    return img


def geo_transform():
    img = image()
    # basic geometric transformation
    tform = transform.SimilarityTransform(scale=1, rotation=math.pi/4,
                                    translation=(img.shape[0]/2, -100))
    print(tform.params)
    
    # write transformations/ get tranformation matrices which rotate the image without cutting bits out
    rotated  = transform.warp(img, tform)
    back_rotated = transform.warp(img, tform.inverse)

    fig, ax = plt.subplots(nrows=3)

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[1].imshow(rotated, cmap=plt.cm.gray)
    ax[2].imshow(back_rotated, cmap=plt.cm.gray)

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()

geo_transform()



geo_transform()


test_image = image()

class GeoTransform(object):
    #https://scikit-image.org/docs/stable/auto_examples/transform/plot_geometric.html
    """Perform geometric transofation on a given image.
    Args:
        
    """
    def __init__(self):
        # set the format of input argument for initialisation
        pass

    def __call__(self):
        # set the format of input argument for called object
        pass

    def geo_transform(scale, rotation, translation):
        tform = transform.SimilarityTransform(scale=1, rotation=math.pi/2,
                                      translation=(0, 1))
    print(tform.params)
    pass



def crop_whitespace():
    pass

