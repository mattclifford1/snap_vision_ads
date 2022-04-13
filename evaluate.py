from argparse import ArgumentParser
import os

from data import download, resize_dataset
from data_loader.load import get_database
from wrangling.database_creator import contruct_database
from models import simple, toy_network, network, FaceNet, utils
from evaluation import eval_torch_model


def eval(model):
    # get the csv of the image paths of the evaluation database
    data = get_database(eval=True)
    # loop over the evaluation database
    for row in data:
        im_path = row['image_path']
        im_label = row['label']
        # get embedding for the image
        embedding = model.get_embedding(im_path)


def run_pipeline(ARGS):
    print('Running Pipline with args: ', ARGS)

    # download and upzip dataset if not found
    data_dir = download.get_if_doesnt_exist(ARGS.dataset_dir)
    # downscale the dataset if required
    if not ARGS.big_ims:
        data_dir = resize_dataset.run(data_dir, 512)

    # contruct database from uob_image_set dataset
    contruct_database(data_dir)

    # eval chosen models from input arguments
    if 'simple' in ARGS.models_list:
        eval(simple)

    # if 'simple_net' in ARGS.models_list:
    #     print('\nRunning simple neural network with triplet loss')
    #     input_size = 256
    #     embedding_dims = 64
    #     model = toy_network.toy_network(input_size, embedding_dims)
    #     train_network(model, input_size, ARGS)
    #
    # if 'big_net' in ARGS.models_list:
    #     print('\nRunning big neural network with triplet loss')
    #     input_size = 512
    #     embedding_dims = 128
    #     model = network.network(input_size, embedding_dims)
    #     train_network(model, input_size, ARGS)
    #
    # if 'facenet' in ARGS.models_list:
    #     print('\nRunning FaceNetInception with triplet loss')
    #     input_size = 224
    #     embedding_dims = 128
    #     model = FaceNet.FaceNetInception(input_size, embedding_dims)
    #     train_network(model, input_size, ARGS)


if __name__ == '__main__':
    # get command line arguments
    parser = ArgumentParser(description='Data pipeline for training and evaluating image embeddings')
    parser.add_argument("--dataset_dir", default='data', help='Location to read/save the uob_image_set used to training/eval')
    parser.add_argument("--big_ims", default=False, action='store_true', help='use full size images for training')
    parser.add_argument("-m", "--models_list", nargs="+", default='simple', choices=['simple', 'simple_net', 'big_net', 'facenet'], help='list of models to use')
    parser.add_argument("--redo_simple_features", default=False, action='store_true', help='calculate simple image features from scratch rather than database look up')
    parser.add_argument("--batch_size", default=16, type=int, help='batch size to use during training')
    parser.add_argument("--checkpoint", default=None, help='pretained network weights to load')
    ARGS = parser.parse_args()

    run_pipeline(ARGS)
