'''
to be used with pytorch Transformations with a dataset loader

tsfm = Transform(params)
transformed_sample = tsfm(sample)
'''
import numpy as np
import torch
from torchvision import transforms
from skimage import transform

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
