'''
compare to nearest embeddings and see if they are of the same class
'''
from torch.utils.data import Dataset, DataLoader
import multiprocessing
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import sys
sys.path.append('.')
sys.path.append('..')
from data_loader.load import get_data
from data_loader.augmentation import *


class eval:
    def __init__(self, embeddings, labels, num_neighbours=3, compute_sequencially=False):
        self.embeddings = embeddings   # list
        self.labels = labels           # list
        self.num_neighbours = num_neighbours
        self.compute_sequencially = compute_sequencially
        self.compute()

    def compute(self):
        if self.compute_sequencially:
            scores = self.run_sequentially()
        else:
            scores = self.run_parellel()
        self.accuracy = (sum(scores)/len(scores))
        print('Accuracy: ', self.accuracy*100, '%')

    def run_sequentially(self):
        scores = []
        for i in range(len(self.labels)):
            scores.append(self.get_score(i))
        return scores

    def run_parellel(self):
        pool_obj = multiprocessing.Pool()
        scores = list(tqdm(pool_obj.imap(self.get_score, range(len(self.labels))), total=len(self.labels)))
        return scores

    def get_score(self, i):
        X_train = list(self.embeddings) # copy
        y_train = list(self.labels) # copy
        x_test = X_train.pop(i)
        y_test = y_train.pop(i)
        y_pred = get_knn_class(X_train, y_train, [x_test], self.num_neighbours)
        if y_pred[0] == y_test:
            return 1
        else:
            return 0


def get_knn_class(X_train, y_train, x_test, num_neighbours=3):
    neigh = KNeighborsClassifier(num_neighbours)
    neigh.fit(X_train, y_train)
    return neigh.predict(x_test)


def eval_torch_model(model, device='cuda', csv='wrangling/image_paths.csv'):
    batch_size = 16
    device = torch.device(device)
    trans = transforms.Compose([Rescale((model.input_size, model.input_size)),
                                ToTensor()])
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
    embeddings = [[1,1], [2,4], [5,4], [2,1], [1,1]]
    labels = ['a', 'b', 'b', 'a', 'a']
    # eval(embeddings, labels)
    import torch
    from models.toy_network import Network
    model = Network(512, 16)
    model.load_state_dict(torch.load('data/files_to_gitignore/trained_simple_model1.pth'))
    model.eval()
    eval_torch_model(model)
