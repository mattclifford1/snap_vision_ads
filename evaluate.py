from argparse import ArgumentParser
import os
from tqdm import tqdm
from skimage import io

from data import download, resize_dataset
from data_loader.load import get_database
from wrangling.database_creator import contruct_database
from models import simple, toy_network, network, FaceNet, utils
from evaluation import nearest_points


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

    # now find images of the closest embeddings
    for i in tqdm(range(len(data_results['image_paths'])), desc="Getting knn closest embeddings", leave=False):
        closest = nearest_points.get_knn_closest(data_results['embeddings'],
                                                 data_results['image_paths'],
                                                 [data_results['embeddings'][i]],
                                                 num_neighbours=5)
        data_results['closest'].append(closest)

    # now do whatever you want with the images and their closest embeddings
    count = 0
    for i in tqdm(range(len(data_results['image_paths'])), desc="Evaluating closest embeddings", leave=False):
        # read images from file
        image_path = data_results['image_paths'][i]
        image = io.imread(image_path)
        closest = []
        for closest_image__path in data_results['closest'][i]:
            closest.append(io.imread(closest_image__path))
        # now maybe plot the image and its closest?
        # WRITE CODE HERE
        #
        #
        #

        # don't loop over the whole dataset whilst developing
        if count > 10:
            break
        count += 1



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
        simple_model = simple.model(os.path.join(ARGS.save_dir, 'simple_model.csv'))
        results = simple_model.run()
        eval(simple_model)

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
    parser.add_argument("--save_dir", default='data/files_to_gitignore/models', help='Location models were saved during training')
    parser.add_argument("--batch_size", default=16, type=int, help='batch size used during training')
    parser.add_argument("-lr", "--learning_rate", default=0.001, type=float, help='learning rate used during training')
    parser.add_argument("-lrd", "--lr_decay", default=0.95, type=float, help='learning rate dacay used during training')
    parser.add_argument("--checkpoint", default=None, help='pretained network weights to load')
    ARGS = parser.parse_args()

    run_pipeline(ARGS)
