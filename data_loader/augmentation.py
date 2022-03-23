'''
to be used with pytorch Transformations with a dataset loader

tsfm = Transform(params)
transformed_sample = tsfm(sample)
'''
import numpy as np
from torchvision import transforms, utils
from skimage import transform

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple): Desired output size. Output is
            matched to output_size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        image, positive, negative = sample['image'], sample['positive'], sample['negative']
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        pos = transform.resize(positive, (new_h, new_w))
        neg = transform.resize(negative, (new_h, new_w))
        return {'image': img, 'positive': pos, 'negative': neg}


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
        image, positive, negative = sample['image'], sample['positive'], sample['negative']
        image = self.crop(image)
        positive = self.crop(positive)
        negative = self.crop(negative)
        return {'image': image, 'positive': positive, 'negative': negative}

    def crop(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h,
                      left: left + new_w]
        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, positive, negative = sample['image'], sample['positive'], sample['negative']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        positive = positive.transpose((2, 0, 1))
        negative = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'positive': torch.from_numpy(positive),
                'negative': torch.from_numpy(negative)}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    sys.path.append('.')
    sys.path.append('..')
    from data_loader.load import get_training_data
    training_data = get_training_data()

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
