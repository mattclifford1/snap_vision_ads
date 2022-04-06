'''
simple toy model using non-deep embeddings
'''
import os
import pandas as pd
import ast
import sys
sys.path.append('.')
sys.path.append('..')
from evaluation import compare_similar


def run():
    # check features file exists (create if not)
    features_csv = 'exploration/database.csv'
    if not os.path.isfile(features_csv):
        from exploration import save_features
        save_features.save_simple_features()

    # load features
    df = pd.read_csv(features_csv)
    embeddings = []
    # get all features from dataframe
    for row in range(df.shape[0]):
        # convert dict in str format to list of its values
        dict_row = ast.literal_eval(df['features'][row])
        embeddings.append(list(dict_row.values()))
    labels = list(df['label'])

    # eval features
    evaluation = compare_similar.eval(embeddings, labels)  # will print out accuracy


if __name__ == '__main__':
    run()
