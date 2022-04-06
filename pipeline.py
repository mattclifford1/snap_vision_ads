from argparse import ArgumentParser
import os

from wrangling.database_creator import contruct_database
from wrangling.explore_dataset import print_stats
from models import simple, toy_network
from training import torch_trainer
from evaluation import eval_torch_model

if __name__ == '__main__':
    # get command line arguments
    parser = ArgumentParser(description='Data pipeline for training and evaluating image embeddings')
    parser.add_argument("--dataset_dir", default='data/uob_image_set', help='Location to read/save the uob_image_set used to training/eval')
    parser.add_argument("--dataset_stats", default=False, action='store_true', help='prints out some basic statistics about the dataset')
    parser.add_argument("-m", "--models_list", nargs="+", default='simple', choices=['simple', 'triplet_simple_net'], help='list of models to use')
    parser.add_argument("--redo_simple_features", default=False, action='store_true', help='calculate simple image features from scratch rather than database look up')
    parser.add_argument("--train", default=False, action='store_true', help='option to train the neural network')
    parser.add_argument("--epochs", default=1, type=int, help='how many epochs to train for')
    parser.add_argument("--checkpoint", default='data/files_to_gitignore/trained_toy_network_epoch_5.pth', help='how many epochs to train for')
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
        input_size = 256
        embedding_dims = 64
        model = toy_network.toy_network(input_size, embedding_dims)
        if ARGS.train:
            print('Training')
            model = torch_trainer.run(model, input_size, ARGS.epochs)
        else:
            toy_network.load_weights(model, ARGS.checkpoint)
        eval_torch_model.run(model)
        print('evaluation')
