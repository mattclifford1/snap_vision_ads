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
from data_loader.load import get_training_data
from data_loader.augmentation import *


class eval:
    def __init__(self, embeddings, labels, num_neighbours=3, compute_sequencially=False):
        self.embeddings = embeddings   # list
        self.labels = labels           # list
        self.num_neighbours = num_neighbours
        self.compute_sequencially = compute_sequencially
        self.compute()

    def compute(self):
        print('Running evaluation')
        if self.compute_sequencially:
            scores = self.run_sequentially()
        else:
            scores = self.run_parellel()
        self.accuracy = (sum(scores)/len(scores))
        print('Accuracy: ', self.accuracy*100, '%')

    def run_sequentially(self):
        scores = []
        for i in tqdm(range(len(self.labels))):
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
        y_pred = self.get_knn_class(X_train, y_train, [x_test])
        if y_pred[0] == y_test:
            return 1
        else:
            return 0

    def get_knn_class(self, X_train, y_train, x_test):
        neigh = KNeighborsClassifier(self.num_neighbours)
        neigh.fit(X_train, y_train)
        return neigh.predict(x_test)


def eval_torch_model(model, csv='wrangling/image_paths.csv'):
    # if model == str then load model
    trans = transforms.Compose([Rescale((512, 512)),
                                ToTensor()])
    transformed_dataset = get_training_data(transform=trans)
    dataloader = DataLoader(transformed_dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=4)
    embeddings = []
    labels = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.script(model).to(device)
    count = 0
    for sample in tqdm(dataloader):
        im = sample['image'].to(device=device, dtype=torch.float)
        embedding = model(im)
        #convert to numpy array
        emeddings_array = embedding.cpu().detach().numpy()

        embeddings.append(list(emeddings_array[0,:]))
        labels.append(str(sample['label'].cpu().detach().numpy()))
        # if count >10:
        #     break
        count +=1
    print(embeddings)
    print(labels)
    eval(embeddings, labels, compute_sequencially=True)


if __name__ == '__main__':
    embeddings = [[1,1], [2,4], [5,4], [2,1], [1,1]]
    labels = ['a', 'b', 'b', 'a', 'a']
    # eval(embeddings, labels)
    import torch
    from training.trainer import Network
    model = Network(512, 16)
    # model.load_state_dict(torch.load('data/files_to_gitignore/trained_simple_model.pth'))
    # model.eval()
    eval_torch_model(model)
