from argparse import ArgumentParser
import os

from wrangling.database_creator import contruct_database
from wrangling.explore_dataset import print_stats
from models import simple, toy_network, network, FaceNet, utils
from training import torch_trainer
from evaluation import eval_torch_model

def print_results(results):
    for i in range(len(results['closest'])):
        print('Closest '+str(i+1)+': ', results['closest'][i]*100, '%')
    print('Any closest embedding correct: '+str(results['in_any']*100))

def train_network(model, input_size, ARGS):
    if ARGS.train:
        print('Training')
        model = torch_trainer.run(model, input_size, ARGS.epochs, ARGS.batch_size, ARGS.save_dir)
    else:
        utils.load_weights(model, ARGS.checkpoint)
    print('evaluation')
    results = eval_torch_model.run(model, batch_size=ARGS.batch_size)
    print_results(results)


if __name__ == '__main__':
    # get command line arguments
    parser = ArgumentParser(description='Data pipeline for training and evaluating image embeddings')
    parser.add_argument("--dataset_dir", default='data/uob_image_set', help='Location to read/save the uob_image_set used to training/eval')
    parser.add_argument("--dataset_stats", default=False, action='store_true', help='prints out some basic statistics about the dataset')
    parser.add_argument("-m", "--models_list", nargs="+", default='simple', choices=['simple', 'simple_net', 'big_net', 'facenet'], help='list of models to use')
    parser.add_argument("--redo_simple_features", default=False, action='store_true', help='calculate simple image features from scratch rather than database look up')
    parser.add_argument("--train", default=False, action='store_true', help='option to train the neural network')
    parser.add_argument("--save_dir", default='data/files_to_gitignore', help='Location to save models during training')
    parser.add_argument("--epochs", default=1, type=int, help='how many epochs to train for')
    parser.add_argument("--batch_size", default=16, type=int, help='batch size to use during training')
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
        results = simple.run(features_csv)
        print_results(results)

    if 'simple_net' in ARGS.models_list:
        print('\nRunning simple neural network with triplet loss')
        input_size = 256
        embedding_dims = 64
        model = toy_network.toy_network(input_size, embedding_dims)
        train_network(model, input_size, ARGS)

    if 'big_net' in ARGS.models_list:
        print('\nRunning big neural network with triplet loss')
        input_size = 1024
        embedding_dims = 128
        model = network.network(input_size, embedding_dims)
        train_network(model, input_size, ARGS)

    if 'facenet' in ARGS.models_list:
        print('\nRunning FaceNetInception with triplet loss')
        input_size = 224
        embedding_dims = 128
        model = FaceNet.FaceNetInception(embedding_dims)
        train_network(model, input_size, ARGS)
