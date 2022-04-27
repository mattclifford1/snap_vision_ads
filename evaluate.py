from argparse import ArgumentParser
import os
from tqdm import tqdm
from skimage import io
import matplotlib.pyplot as plt

from data import download, resize_dataset
from data_loader.load import get_database
from wrangling.database_creator import contruct_database
from models import simple, toy_network, network, FaceNet, neural_network
from evaluation import nearest_points

def get_label(filename):
    head, _ = os.path.split(filename)
    return int(os.path.basename(head))


def plot_closest(image_path, closest_paths, row, rows, count):
    # plot the image and its closest
    row_num = (row-1)*(len(closest_paths)+1)
    ax = plt.subplot(rows, len(closest_paths)+1, row_num+1)
    plt.imshow(io.imread(image_path))
    if row == True:
        ax.set_title('Input Image')
    plt.yticks([])
    plt.xticks([])
    plt.xlabel(get_label(image_path))
    plt.ylabel(count, rotation=0)
    count = 1
    for close_path in closest_paths:
        ax = plt.subplot(rows, len(closest_paths)+1, row_num+count+1)
        plt.imshow(io.imread(close_path))
        if row == True:
            ax.set_title(str(count))
        plt.yticks([])
        plt.xticks([])
        plt.xlabel(get_label(close_path))
        count += 1


def eval(model, ARGS):
    # get the csv of the image paths of the evaluation database
    data = get_database(eval=True)

    # loop over the evaluation database to get the data we need for evaluation
    data_results = {'image_paths': [],
                    'labels': [],
                    'embeddings': [],
                    'closest': [],
                    'closest_labels': []}
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
        closest_labels = [get_label(x) for x in closest]

        data_results['closest'].append(closest)
        data_results['closest_labels'].append(closest_labels)

    # now do whatever you want with the images and their closest embeddings
    row_count = 1
    count = 0
    fig = plt.figure()
    for i in tqdm(range(len(data_results['image_paths'])), desc="Evaluating closest embeddings", leave=False):
        count += 1
        # deterime if this embedding has passed or failed on labels
        if data_results['labels'][i] in data_results['closest_labels'][i]: # passed
            if 'pass' not in ARGS.show_case:
                continue
        else:
            if 'fail' not in ARGS.show_case:
                continue

        image_path = data_results['image_paths'][i]
        closest_paths = data_results['closest'][i]
        plot_closest(image_path, closest_paths, row_count, ARGS.num_disp, count)
        if row_count >= ARGS.num_disp:
            plt.show()
            row_count = 1
            # break ## REMOVE THIS TO SHOW MULTIPLE BATCHES OF IMAGES
            fig = plt.figure()
        else:
            row_count += 1



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
        eval(simple_model, ARGS)

    if 'simple_net' in ARGS.models_list:
        model = toy_network.toy_network()
        net_model = neural_network.run_net(ARGS.save_dir, model, ARGS.learning_rate, ARGS.lr_decay, ARGS.batch_size, ARGS.checkpoint)
        eval(net_model, ARGS)

    if 'big_net' in ARGS.models_list:
        model = network.network()
        net_model = neural_network.run_net(ARGS.save_dir, model, ARGS.learning_rate, ARGS.lr_decay, ARGS.batch_size, ARGS.checkpoint)
        eval(net_model, ARGS)

    if 'facenet' in ARGS.models_list:
        model = FaceNet.FaceNetInception()
        net_model = neural_network.run_net(ARGS.save_dir, model, ARGS.learning_rate, ARGS.lr_decay, ARGS.batch_size, ARGS.checkpoint)
        eval(net_model, ARGS)


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
    # what type of results to show
    parser.add_argument("-sc", "--show_case", nargs="+", default=['pass', 'fail'], choices=['pass', 'fail'], help='list of eval types to display')
    parser.add_argument("--num_disp", default=1, type=int, help='number of eval example to display at once')
    ARGS = parser.parse_args()

    run_pipeline(ARGS)
