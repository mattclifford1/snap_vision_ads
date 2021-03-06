from argparse import ArgumentParser
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import download, resize_dataset
from data_loader.load import get_database
from wrangling.database_creator import contruct_database
from models import simple, dom_colours, toy_network, network, FaceNet, neural_network
from evaluation import nearest_points, colour_compare
from utils import get_label, plot_closest

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
                                                 num_neighbours=ARGS.num_neighbours)
        closest_labels = [get_label(x) for x in closest]

        data_results['closest'].append(closest)
        data_results['closest_labels'].append(closest_labels)

    # now do whatever you want with the images and their closest embeddings
    row_count = 1
    count = 0
    pic_size = 1
    fig = plt.figure(figsize=(pic_size*ARGS.num_neighbours, pic_size*ARGS.num_disp))
    for i in tqdm(range(len(data_results['image_paths'])), desc="Evaluating closest embeddings", leave=False):
        count += 1
        if ARGS.show_nums != []:
            if str(count) not in ARGS.show_nums:
                continue
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
            if ARGS.save_fig:
                save_dir = os.path.join('data', 'files_to_gitignore', 'eval_figs', str(ARGS.model)+str(ARGS.show_case))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_file = os.path.join(save_dir, str(count)+'.png')
                plt.savefig(save_file, bbox_inches='tight')
            else:
                plt.show()
            row_count = 1
            # break ## REMOVE THIS TO SHOW MULTIPLE BATCHES OF IMAGES
            fig = plt.figure(figsize=(pic_size*ARGS.num_neighbours, pic_size*ARGS.num_disp))
        else:
            row_count += 1
    # final fig
    if ARGS.save_fig:
        save_dir = os.path.join('data', 'files_to_gitignore', 'eval_figs', str(ARGS.model)+str(ARGS.show_case))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_file = os.path.join(save_dir, str(count)+'.png')
        plt.savefig(save_file, bbox_inches='tight')
    else:
        plt.show()


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
    if 'simple' == ARGS.model:
        simple_model = simple.model(os.path.join(ARGS.save_dir, 'simple_model.csv'))
        eval(simple_model, ARGS)

    elif 'dom_colours' == ARGS.model:
        dom_colours_model = dom_colours.model(os.path.join(ARGS.save_dir, 'dom_colours_model.csv'))
        eval(dom_colours_model, ARGS)

    elif 'simple_net' == ARGS.model:
        model = toy_network.toy_network()
        net_model = neural_network.run_net(ARGS.save_dir, model, ARGS.learning_rate, ARGS.lr_decay, ARGS.batch_size, ARGS.checkpoint)
        eval(net_model, ARGS)

    elif 'big_net' == ARGS.model:
        model = network.network()
        net_model = neural_network.run_net(ARGS.save_dir, model, ARGS.learning_rate, ARGS.lr_decay, ARGS.batch_size, ARGS.checkpoint)
        eval(net_model, ARGS)

    elif 'facenet' == ARGS.model:
        model = FaceNet.FaceNetInception()
        net_model = neural_network.run_net(ARGS.save_dir, model, ARGS.learning_rate, ARGS.lr_decay, ARGS.batch_size, ARGS.checkpoint)
        eval(net_model, ARGS)


if __name__ == '__main__':
    # get command line arguments
    parser = ArgumentParser(description='Data pipeline for training and evaluating image embeddings')
    parser.add_argument("--dataset_dir", default='data', help='Location to read/save the uob_image_set used to training/eval')
    parser.add_argument("--big_ims", default=False, action='store_true', help='use full size images for training')
    parser.add_argument("-m", "--model", type=str, default='simple', choices=['simple', 'dom_colours', 'simple_net', 'big_net', 'facenet'], help='models to use')
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
    parser.add_argument("--num_neighbours", default=5, type=int, help='number of closest neighbours to display')
    parser.add_argument("--save_fig", default=False, action='store_true', help='save figure of closest embeddings to file')
    parser.add_argument("--show_nums", nargs="+", default=[], help='show specific samples from eval set')
    ARGS = parser.parse_args()

    run_pipeline(ARGS)
