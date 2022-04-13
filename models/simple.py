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


def get_features_csv(features_csv='exploration/database.csv'):
    # check features file exists (create if not)
    if not os.path.isfile(features_csv):
        from exploration import save_features
        save_features.save_simple_features(features_csv=features_csv)
    df = pd.read_csv(features_csv)
    return df

def run(features_csv='exploration/database.csv'):
    '''
    run evaluation on all images
    '''
    # load features
    df = get_features_csv(features_csv)
    embeddings = []
    # get all features from dataframe
    for row in range(df.shape[0]):
        # convert dict in str format to list of its values
        dict_row = ast.literal_eval(df['features'][row])
        embeddings.append(list(dict_row.values()))
    labels = list(df['label'])
    # eval features
    evaluation = compare_similar.eval(embeddings, labels)  # will print out accuracy
    return evaluation.results

def get_embedding(image_path, features_csv='exploration/database.csv'):
    # load features
    df = get_features_csv(features_csv)
    row = df[df['image_path'] == image_path].index
    if len(row) == 0:
        raise Exception('Image path: '+str(image_path)+ ' not in '+str(features_csv)+'\nMaybe you need to generate features first?')
    elif len(row) > 1:
        raise Exception('More than one instance of: '+str(image_path)+ ' found in '+str(features_csv))
    else:
        row = row[0]
    dict_row = ast.literal_eval(df['features'][row])
    embedding = list(dict_row.values())
    return embedding


if __name__ == '__main__':
    # run()
    embedding = get_embedding('data/uob_image_set512/13507603/13507603_2.jpg')
    print(embedding)
