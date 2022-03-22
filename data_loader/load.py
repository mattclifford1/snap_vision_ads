import pandas as pd
import os
from skimage import io
import random
import ast


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
    def __init__(self, df_csv='wrangling/image_paths.csv'):
        self.df_csv = df_csv
        self.df = pd.read_csv(self.df_csv)
        self.image_paths = self.df['image_path'].tolist()
        self.similar_images = self.df['similar_images'].tolist()
        # convert str of similar images to lists
        for i in range(len(self.similar_images)):
            self.similar_images[i] = ast.literal_eval(self.similar_images[i])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        # get file paths for im, pos, neg example
        image_path = self.image_paths[i]
        random_similar = random.choice(self.similar_images[i])
        not_similars = list(set(self.image_paths) - set(self.similar_images[i]))
        random_not_similar = random.choice(not_similars)
        # read images from disk
        image_path = io.imread(image_path)
        random_similar = io.imread(random_similar)
        random_not_similar = io.imread(random_not_similar)
        return {'image':    image_path,
                'positive': random_similar,
                'negative': random_not_similar}


if __name__ == '__main__':
    data = get_database()
    # print(len(data))
    # print(data[0])
    training_data = get_training_data()
    print(training_data[0])
