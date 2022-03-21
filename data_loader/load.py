import pandas as pd
import os


class get_database:
    def __init__(self, df_csv='wrangling/image_paths.csv'):
        self.df_csv = df_csv
        self.check_csv_exists()
        self.load_df()

    def check_csv_exists(self):
        if not os.path.exists(self.df_csv):
            raise FileNotFoundError('No database file found: '+self.df_csv)

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
        # return {'image_path': row['image_path'],
        #         'similar_paths': row['similar_images']}



if __name__ == '__main__':
    data = get_database()
    print(len(data))
    print(data[0])
