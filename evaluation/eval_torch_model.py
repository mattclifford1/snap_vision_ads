from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('.')
sys.path.append('..')
from data_loader.load import get_data
from data_loader.augmentation import *
from evaluation.compare_similar import eval


def run(model,
        device='cuda',
        csv='wrangling/image_paths_database_eval.csv'):
    model.eval()
    batch_size = 16
    device = torch.device(device)
    trans = transforms.Compose([Rescale((model.input_size, model.input_size))])
    transformed_dataset = get_data(transform=trans, eval=True)
    dataloader = DataLoader(transformed_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=12,
                            prefetch_factor=1)
    embeddings = []
    labels = []
    model = model.to(device)
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
        # if count >10:
        #     break
        count +=1
    evaller = eval(embeddings, labels, compute_sequencially=True)
    return evaller.accuracy*100


if __name__ == '__main__':
    import torch
    from models.toy_network import Network
    model = Network(512, 16)
    model.load_state_dict(torch.load('data/files_to_gitignore/trained_simple_model1.pth'))
    model.eval()
    run(model)
