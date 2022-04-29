'''
simple toy model using non-deep embeddings
'''
import os
import pandas as pd
import ast
import sys
sys.path.append('.')
sys.path.append('..')
from evaluation import nearest_points


def get_features_csv(features_csv, apply_mask=False):
    # don't check features file exists if using masked images
    if apply_mask == True:
        from exploration import save_features
        save_features.save_simple_features(features_csv=features_csv, apply_mask=True)
    # check features file exists (create if not)
    if not os.path.isfile(features_csv):
        from exploration import save_features
        save_features.save_simple_features(features_csv=features_csv)
    df = pd.read_csv(features_csv)
    return df


class model:
    def __init__(self, csv_file, apply_mask=False):
        self.features_csv = csv_file
        self.apply_mask = apply_mask
        if self.apply_mask == True:
            self.df = get_features_csv(self.features_csv, self.apply_mask)
        else:
            self.df = get_features_csv(self.features_csv)

    def run(self):
        '''
        run evaluation on all images
        '''
        # load features
        embeddings = []
        # get all features from dataframe
        for row in range(self.df.shape[0]):
            # convert dict in str format to list of its values
            dict_row = ast.literal_eval(self.df['features'][row])
            embeddings.append(list(dict_row.values()))
        labels = list(self.df['label'])
        # eval features
        evaluation = nearest_points.eval(embeddings, labels)  # will print out accuracy
        return evaluation.results

    def get_embedding(self, image_path):
        '''
        get embedding of a single image
        '''
        # load features
        row = self.df[self.df['image_path'] == image_path].index
        if len(row) == 0:
            raise Exception('Image path: '+str(image_path)+ ' not in '+str(self.features_csv)+'\nMaybe you need to generate features first?')
        elif len(row) > 1:
            raise Exception('More than one instance of: '+str(image_path)+ ' found in '+str(features_csv))
        else:
            row = row[0]
        dict_row = ast.literal_eval(self.df['features'][row])
        embedding = list(dict_row.values())
        return embedding


if __name__ == '__main__':
    # run()
    m = model('data/files_to_gitignore/models/simple_model.csv')
    embedding = m.get_embedding('data/uob_image_set512/16288974/16288974_1.jpg')
    print(embedding)
