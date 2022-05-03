'''
to be used with pytorch Transformations with a dataset loader

tsfm = Transform(params)
transformed_sample = tsfm(sample)
'''
import numpy as np
import torch
from torchvision import transforms
from skimage import transform

import cv2 # add to requirements

class Rescale(object): # and read
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple): Desired output size. Output is
            matched to output_size.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size
        new_h, new_w = self.output_size
        self.new_h, self.new_w = int(new_h), int(new_w)

    def __call__(self, sample):
        for key in sample.keys():
            sample[key] = transform.resize(sample[key], (self.new_h, self.new_w))
        return sample



class Mask(object):
    """Apply thresholding to image
    
    Args:
        Fixed values for thresholding # edit 
    """

    def __init__(self, threshold_params):
        #assert isinstance(threshold_params, tuple)
        assert isinstance(threshold_params, list)
        #assert len(threshold_params) = 2
        #assert(all(isinstance(item, tuple) for item in threshold_params))
        self.threshold_params = threshold_params
        lower, upper = self.threshold_params
        self.lower, self.upper = np.array(list(lower)), np.array(list(upper))

    def __call__(self, sample):
        for key in sample.keys():
            sample[key] = transform.resize(sample[key], (self.new_h, self.new_w))
        return sample

    def mask(self, image):
        h, w = image.shape[:2]
        # Create mask to only select black 
        thresh = cv2.inRange(image, self.lower, self.upper)
        # apply morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # invert morp image
        mask = 255 - morph
        # apply mask to image
        result = cv2.bitwise_and(image, image, mask=mask)
        return result




class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        for key in sample.keys():
            sample[key] = self.crop(sample[key])
        return sample

    def crop(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h,
                      left: left + new_w]
        return image


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    sys.path.append('.')
    sys.path.append('..')
    from data_loader.load import get_data
    training_data = get_data()

    scale = Rescale((256,256))
    crop = RandomCrop(128)
    composed = transforms.Compose([Rescale((256,256)),
                                   RandomCrop(224)])

    # Apply each of the above transforms on sample.
    fig = plt.figure()
    sample = training_data[65]
    for i, tsfrm in enumerate([scale, crop, composed]):
        transformed_sample = tsfrm(sample)
        ax = plt.subplot(3, 3, i*3 + 1)
        plt.imshow(transformed_sample['image'])
        ax.set_title('im '+ type(tsfrm).__name__)
        ax = plt.subplot(3, 3, i*3 + 2)
        plt.imshow(transformed_sample['positive'])
        ax.set_title('pos '+ type(tsfrm).__name__)
        ax = plt.subplot(3, 3, i*3 + 3)
        plt.imshow(transformed_sample['negative'])
        ax.set_title('neg '+ type(tsfrm).__name__)
        plt.tight_layout()
        ax.axis('off')

    plt.show()
