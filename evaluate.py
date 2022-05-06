from argparse import ArgumentParser
import os
from tqdm import tqdm
from skimage import io

from data import download, resize_dataset
from data_loader.load import get_database
from wrangling.database_creator import contruct_database
from models import simple, toy_network, network, FaceNet, neural_network
import pandas as pd

def eval(model):
    # get the csv of the image paths of the evaluation database
    data = get_database(eval=True)

    # loop over the evaluation database to get the data we need for evaluation
    data_results = {'image_paths': [],
                    'labels': [],
                    'embeddings': [],
                    'closest': []}
    for row in tqdm(data, desc="Getting Embeddings", leave=False):
        data_results['image_paths'].append(row['image_path'])
        data_results['labels'].append(row['label'])
        # get embedding for the image
        embedding = model.get_embedding(row['image_path'])
        data_results['embeddings'].append(embedding)

    df = pd.DataFrame()
    for i in range(64):
        df['feature' + str(i)] = []

    for i in range(len(data_results['embeddings'])):
        df.loc[i] = data_results['embeddings'][i]

    df.to_csv('embeddings.csv', index=False)

def run_pipeline(ARGS):
    print('Running Pipline with args: ', ARGS)

    # download and upzip dataset if not found
    data_dir = download.get_if_doesnt_exist(ARGS.dataset_dir)
    # downscale the dataset if required
    if not ARGS.big_ims:
        data_dir = resize_dataset.run(data_dir, 512)

    # contruct database from uob_image_set dataset
    contruct_database(data_dir)

    # download network weights if required
    if ARGS.download_weights:
        print('Download model weights')
        download._download_and_unzip(ARGS.save_dir, ARGS.weights_url)

    # eval chosen models from input arguments
    if 'simple' in ARGS.models_list:
        simple_model = simple.model(os.path.join(ARGS.save_dir, 'simple_model.csv'))
        eval(simple_model)

    if 'simple_net' in ARGS.models_list:
        model = toy_network.toy_network()
        net_model = neural_network.run_net(ARGS.save_dir, model, ARGS.learning_rate, ARGS.lr_decay, ARGS.batch_size, ARGS.checkpoint)
        eval(net_model)

    if 'big_net' in ARGS.models_list:
        model = network.network()
        net_model = neural_network.run_net(ARGS.save_dir, model, ARGS.learning_rate, ARGS.lr_decay, ARGS.batch_size, ARGS.checkpoint)
        eval(net_model)

    if 'facenet' in ARGS.models_list:
        model = FaceNet.FaceNetInception()
        net_model = neural_network.run_net(ARGS.save_dir, model, ARGS.learning_rate, ARGS.lr_decay, ARGS.batch_size, ARGS.checkpoint)
        eval(net_model)


if __name__ == '__main__':
    # get command line arguments
    parser = ArgumentParser(description='Data pipeline for training and evaluating image embeddings')
    parser.add_argument("--dataset_dir", default='data', help='Location to read/save the uob_image_set used to training/eval')
    parser.add_argument("--big_ims", default=False, action='store_true', help='use full size images for training')
    parser.add_argument("-m", "--models_list", nargs="+", default='simple', choices=['simple', 'simple_net', 'big_net', 'facenet'], help='list of models to use')
    # select model weights (if neural network model)
    parser.add_argument("--save_dir", default='data/files_to_gitignore/models', help='Location models were saved during training')
    parser.add_argument("--checkpoint", default='latest', help='epoch of pretained network weights to load')
    parser.add_argument("--download_weights", default=False, action='store_true', help='download network weights')
    parser.add_argument("--weights_url", default='https://drive.google.com/u/0/uc?id=1bTp1vVcAyi4dhTxr_WGKPqTVPrTIM0gB&export=download&confirm=t', help='url of network weights to download')
    # training params are to know which model to load
    parser.add_argument("--batch_size", default=64, type=int, help='batch size used during training')
    parser.add_argument("-lr", "--learning_rate", default=0.0001, type=float, help='learning rate used during training')
    parser.add_argument("-lrd", "--lr_decay", default=0.95, type=float, help='learning rate dacay used during training')
    ARGS = parser.parse_args()

    run_pipeline(ARGS)
