import pandas as pd
import os
from skimage import io
import torch
import random
import ast
import sys
sys.path.append('.')
sys.path.append('..')
from data_loader.augmentation import *


class get_database:
    def __init__(self, df_csv='wrangling/image_paths.csv'):
        self.df_csv = df_csv
        self.check_csv_exists()
        self.load_df()

    def check_csv_exists(self):
        if not os.path.exists(self.df_csv):
            # raise FileNotFoundError('No database file found: '+self.df_csv)
            # create database instead
            import sys
            sys.path.append('.')
            from wrangling import database_creator
            database_creator.contruct_database()

    def load_df(self):
        self.df = pd.read_csv(self.df_csv)

    def add_col(self, data, col_name):
        self.df[col_name] = data

    def save_df(self, save_file):
        self.df.to_csv(save_file, index=False)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        row = self.df.iloc[i]
        return row



'''
to be used with torch's dataloader when training with triplet loss
'''
class get_training_data:
    def __init__(self,
                 df_csv='wrangling/image_paths.csv',
                 transform=None):
        self.df_csv = df_csv
        self.transform = transform
        self.df = pd.read_csv(self.df_csv)
        self.image_paths = self.df['image_path'].tolist()
        self.similar_images = self.df['similar_images'].tolist()
        self.labels = self.df['label'].tolist()
        # convert str of similar images to lists
        for i in range(len(self.similar_images)):
            self.similar_images[i] = ast.literal_eval(self.similar_images[i])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()
        # get file paths for im, pos, neg example
        image_path = self.image_paths[i]
        random_similar = random.choice(self.similar_images[i])
        not_similars = list(set(self.image_paths) - set(self.similar_images[i]))
        random_not_similar = random.choice(not_similars)
        # read images from disk
        image_path = io.imread(image_path)
        random_similar = io.imread(random_similar)
        random_not_similar = io.imread(random_not_similar)
        # make into sample
        sample = {'image':    image_path,
                  'positive': random_similar,
                  'negative': random_not_similar}
        # data transforms
        if self.transform:
            sample = self.transform(sample)
        # add label to sample
        sample['label'] = self.labels[i]
        return sample


if __name__ == '__main__':

    # plot a few of the training examples
    import matplotlib.pyplot as plt
    training_data = get_training_data()
    fig = plt.figure()

    for i in range(len(training_data)):
        sample = training_data[i]

        print(i, sample['image'].shape, sample['positive'].shape, sample['negative'].shape)
        ax = plt.subplot(4, 3, i*3 + 1)
        plt.imshow(sample['image'])
        ax.set_title('im')
        ax = plt.subplot(4, 3, i*3 + 2)
        plt.imshow(sample['positive'])
        ax.set_title('pos')
        ax = plt.subplot(4, 3, i*3 + 3)
        plt.imshow(sample['negative'])
        ax.set_title('neg')
        plt.tight_layout()
        ax.axis('off')

        if i == 3:
            plt.show()
            break
