import pandas as pd
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import sys
sys.path.append('..')
sys.path.append('.')
from exploration.features import get_features
from data_loader.load import get_database


def get_all_features(csv_file):
    data = get_database()
    features = []
    for row in tqdm(data):
        image_path = row['image_path']
        features_dict = get_features(image_path)
        features.append(features_dict)

    data.add_col(features, 'features')
    data.save_df('exploration/database.csv')


if __name__ == '__main__':
    # set up command line arguments to specifiy where the dataset is
    parser = ArgumentParser()
    parser.add_argument("--csv_file", default='image_paths.csv')
    ARGS = parser.parse_args()

    get_all_features(ARGS.csv_file)
