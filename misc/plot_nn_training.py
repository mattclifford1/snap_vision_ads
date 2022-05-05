import pandas as pd
import matplotlib.pyplot as plt

def plot(csv, name, col):
    df = pd.read_csv(csv)
    x = df['epoch'].to_list()
    y = df[col].to_list()
    plt.plot(x, y, label=name)



csv1 = 'data/files_to_gitignore/models/FaceNetInception_LR_0.0001_decay_0.98_BS_64/training_stats.csv'
csv2 = 'data/files_to_gitignore/models/network_LR_0.0001_decay_0.95_BS_64/training_stats.csv'
csv3 = 'data/files_to_gitignore/models/toy_network_LR_0.0001_decay_0.95_BS_64/training_stats.csv'
name1 = 'FaceNet Inception'
name2 = 'Bigger Vanilla Network'
name3 = 'Small Vanilla Network'
csvs = [csv1, csv2, csv3]
names = [name1, name2, name3]



for csv, name in zip(csvs, names):
    plot(csv, name, 'mean training loss')
plt.xlabel('Epochs')
plt.ylabel('mean training loss')
plt.legend()
plt.show()

for csv, name in zip(csvs, names):
    plot(csv, name, 'evaluation score')
plt.xlabel('Epochs')
plt.ylabel('evaluation score')
plt.legend()
plt.show()
