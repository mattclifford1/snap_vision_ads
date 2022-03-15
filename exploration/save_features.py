import pandas as pd
import numpy as np
from argparse import ArgumentParser
from exploration.features import get_features

def get_all_features(csv_file):
    df = pd.read_csv(csv_file)
    new_features = []
    for index, row in df.iterrows():
        image_path = row['image_path']
        # calculate new feauters
        features_dict = get_features(image_path)
        # save features
        new_features.append(list(features_dict.values()))
    # add new features to df
    new_features = np.array(new_features)
    col = 0
    for key in features_dict.keys():
        df['key'] = new_features[:, col]
        col += 1
    print(df.head())





if __name__ == '__main__':
    # set up command line arguments to specifiy where the dataset is
    parser = ArgumentParser()
    parser.add_argument("--csv_file", default='image_paths.csv')
    ARGS = parser.parse_args()

    get_all_features(ARGS.csv_file)
