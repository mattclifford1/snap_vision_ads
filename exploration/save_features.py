import pandas as pd
from argparse import ArgumentParser
import features

def get_all_features(csv_file):
    database = pd.read_csv(csv_file)
    print(database.head())



if __name__ == '__main__':
    # set up command line arguments to specifiy where the dataset is
    parser = ArgumentParser()
    parser.add_argument("--csv_file", default='image_paths.csv')
    ARGS = parser.parse_args()

    get_all_features(ARGS.csv_file)
