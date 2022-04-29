from argparse import ArgumentParser
import os

from data import download, resize_dataset
from wrangling.database_creator import contruct_database
from wrangling.explore_dataset import print_stats
from models import simple, toy_network, network, FaceNet, utils
from training import torch_trainer
from evaluation import eval_torch_model


def print_results(results):
    for i in range(len(results['closest'])):
        print('Closest '+str(i+1)+': ', results['closest'][i]*100, '%')
    print('Any closest embedding correct: '+str(results['in_any']*100))


def train_network(model, ARGS):
    # print('Before training Evaluation (Random weights)')
    # results = eval_torch_model.run(model, batch_size=ARGS.batch_size)
    # print_results(results)
    print('Training')
    trainer = torch_trainer.trainer(model,
                                    ARGS.learning_rate,
                                    ARGS.lr_decay,
                                    ARGS.epochs,
                                    ARGS.batch_size,
                                    ARGS.save_dir)
    model = trainer.start()

    print('evaluation')
    results = eval_torch_model.run(model, batch_size=ARGS.batch_size)
    print_results(results)


def run_pipeline(ARGS):
    print('Running Pipline with args: ', ARGS)

    # download and upzip dataset if not found
    data_dir = download.get_if_doesnt_exist(ARGS.dataset_dir)

    # downscale the dataset if required
    if not ARGS.big_ims:
        data_dir = resize_dataset.run(data_dir, 512)

    # contruct database from uob_image_set dataset
    contruct_database(data_dir)

    # print out some basics stats exploring the dataset
    if ARGS.dataset_stats:
        print_stats(data_dir)

    # run models
    if 'simple' in ARGS.models_list:
        simple_model_csv = os.path.join(ARGS.save_dir, 'simple_model.csv')
        if os.path.isfile(simple_model_csv):   # calcuate features from scratch rather than using cached
            os.remove(simple_model_csv)
        print('\nRunning simple model')
        
        # mask images 
        if ARGS.apply_mask:
            simple_model = simple.model(simple_model_csv, apply_mask=True)
        else:
            simple_model = simple.model(simple_model_csv)
        results = simple_model.run()
        print_results(results)

    if 'simple_net' in ARGS.models_list:
        print('\nRunning simple neural network with triplet loss')
        model = toy_network.toy_network()
        train_network(model, ARGS)

    if 'big_net' in ARGS.models_list:
        print('\nRunning big neural network with triplet loss')
        model = network.network()
        train_network(model, ARGS)

    if 'facenet' in ARGS.models_list:
        print('\nRunning FaceNetInception with triplet loss')
        model = FaceNet.FaceNetInception()
        train_network(model, ARGS)


if __name__ == '__main__':
    # get command line arguments
    parser = ArgumentParser(description='Data pipeline for training and evaluating image embeddings')
    parser.add_argument("--dataset_dir", default='data', help='Location to read/save the uob_image_set used to training/eval')
    parser.add_argument("--big_ims", default=False, action='store_true', help='use full size images for training')
    parser.add_argument("--dataset_stats", default=False, action='store_true', help='prints out some basic statistics about the dataset')
    parser.add_argument("-m", "--models_list", nargs="+", default='simple', choices=['simple', 'simple_net', 'big_net', 'facenet'], help='list of models to use')
    parser.add_argument("--save_dir", default='data/files_to_gitignore/models', help='Location to save models during training')
    parser.add_argument("--epochs", default=50, type=int, help='how many epochs to train for')
    parser.add_argument("--batch_size", default=64, type=int, help='batch size to use during training')
    parser.add_argument("-lr", "--learning_rate", default=0.0001, type=float, help='learning rate to use during training')
    parser.add_argument("-lrd", "--lr_decay", default=0.95, type=float, help='learning rate dacay to use during training')
    parser.add_argument("--apply_mask", default = False, help='option to mask images to remove background (experimental)')
    ARGS = parser.parse_args()

    run_pipeline(ARGS)
