from argparse import ArgumentParser
import sys
import multiprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm
sys.path.append('..')
sys.path.append('.')
from exploration.features import get_simple_features
from exploration.features import mask
from data_loader.load import get_database


class save_simple_features:
    def __init__(self, compute_sequencially=False,
                       features_csv='exploration/database.csv',
                       eval=True,
                       apply_mask=False,
                       features_func=get_simple_features):
        self.data = get_database(eval=eval)
        self.im_paths = []
        self.apply_mask = apply_mask
        self.features_func = features_func
        for row in self.data:
            self.im_paths.append(row['image_path'])

        if compute_sequencially:
            features = self.run_sequentially()
        else:
            features = self.run_parellel()

        self.data.add_col(features, 'features')
        self.data.save_df(features_csv)

    def get_feats_func(self, i): # wrapper for multiprocessing func
        return self.features_func(self.im_paths[i], self.apply_mask, lower=[200, 200, 200], upper=[255, 255, 255])

    def run_sequentially(self):
        features = []
        for i in tqdm(range(len(self.im_paths)), desc="Making simple features", leave=False):
            features_dict = self.get_feats_func(i)
            features.append(features_dict)
        return features

    def run_parellel(self):
        pool_obj = multiprocessing.Pool()
        features = list(tqdm(pool_obj.imap(self.get_feats_func, range(len(self.im_paths))), total=len(self.im_paths), desc="Making simple features", leave=False))
        return features


if __name__ == '__main__':
    # calculate and save all features
    save_simple_features(eval=True)
