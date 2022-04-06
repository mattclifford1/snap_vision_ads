from argparse import ArgumentParser
import os

from wrangling.database_creator import contruct_database
from wrangling.explore_dataset import print_stats
from models import simple, toy_network
from training import torch_trainer
from evaluation.compare_similar import eval_torch_model

if __name__ == '__main__':
    # get command line arguments
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", default='data/uob_image_set')
    parser.add_argument("--dataset_stats", default=False, action='store_true')
    parser.add_argument("--redo_simple_features", default=False, action='store_true')
    parser.add_argument("-m", "--models_list", nargs="+", default='simple')
    parser.add_argument("--epochs", default=1, type=int)
    ARGS = parser.parse_args()

    print('Running Pipline with args: ', ARGS)
    # contruct database from uob_image_set dataset
    # it will download and upzip dataset if not found locally
    contruct_database(ARGS.dataset_dir)

    # print out some basics stats exploring the dataset
    if ARGS.dataset_stats:
        print_stats(ARGS.dataset_dir)

    # run models
    if 'simple' in ARGS.models_list:
        features_csv='exploration/database.csv'
        if ARGS.redo_simple_features:   # calcuate features from scratch rather than using cached
            os.remove(features_csv)
        print('\nRunning simple model')
        simple.run(features_csv)

    if 'triplet_simple_net' in ARGS.models_list:
        print('\nRunning simple neural network with triplet loss')
        embedding_dims = 64
        input_size = 256
        model = toy_network.toy_network(input_size, embedding_dims)
        model = torch_trainer.run(model, input_size, ARGS.epochs)
        print('evaluation')
        eval_torch_model(model)