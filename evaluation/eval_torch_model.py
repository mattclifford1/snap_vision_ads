import torch
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('.')
sys.path.append('..')
from data_loader.load import get_data
from data_loader.augmentation import *
from evaluation.compare_similar import eval


def run(model,
        device='cuda',
        csv='wrangling/image_paths_database_eval.csv',
        batch_size=16):
    with torch.no_grad():
        # set up model and dataloader
        model.eval()
        device = torch.device(device)
        model = model.to(device)
        trans = transforms.Compose([Rescale((model.input_size, model.input_size))])
        transformed_dataset = get_data(transform=trans, eval=True)
        dataloader = DataLoader(transformed_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=12,
                                prefetch_factor=1)
        # get embeddings from model
        embeddings = []
        labels = []
        count = 0
        for sample in dataloader:
            im = sample['image'].to(device=device, dtype=torch.float)
            embedding = model(im)
            #convert to numpy array
            embeddings_array = embedding.cpu().detach().numpy()
            labels_array = sample['label'].cpu().detach().numpy()
            for i in range(embeddings_array.shape[0]):  # loop over batch
                embeddings.append(list(embeddings_array[i,:]))
                labels.append(str(labels_array[i]))
            # if count >12:   # uncomment for development purposes
            #     break
            count +=1
        # evaluate embeddings
        evaller = eval(embeddings, labels, compute_sequencially=True)
    return evaller.results


if __name__ == '__main__':
    import torch
    from models import toy_network
    model = toy_network.Network(512, 16)
    toy_network.load_weights(model, 'data/files_to_gitignore/trained_simple_model1.pth')
    model.eval()
    run(model)
