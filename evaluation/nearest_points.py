'''
compare to nearest embeddings and see if they are of the same class
'''
# import multiprocessing
import torch.multiprocessing as multiprocessing
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from tqdm import tqdm

class eval:
    '''
    compare emeddings to labels
    embeddings: list of lists
    labels: list of strs
    '''
    def __init__(self, embeddings, labels, num_neighbours=5, compute_sequencially=False):
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
        scores = np.array(scores) # first col is actual, following cols are closest neighbours
        self.closest = []
        total = []
        for i in range(1, scores.shape[1]):
            same_elements = (scores[:, 0] == scores[:, i])
            total.append(same_elements)
            acc = np.sum(same_elements)/len(same_elements)
            self.closest.append(acc)
        # calcuate if class in any of the nearest embeddings
        total = np.array(total).T
        count = 0
        for i in range(total.shape[0]):
            if True in total[i, :]:
                count += 1
        self.accuracy = count/total.shape[0]
        self.results = {'closest':self.closest,
                        'in_any': self.accuracy}

    def run_sequentially(self):
        scores = []
        for i in tqdm(range(len(self.labels)), desc="Eval Sequential compute", leave=False):
            scores.append(self.get_score(i))
        return scores

    def run_parellel(self):
        pool_obj = multiprocessing.Pool()
        scores = list(tqdm(pool_obj.imap(self.get_score, range(len(self.labels))), total=len(self.labels), desc="Eval Pararrel compute", leave=False))
        return scores

    def get_score(self, i):
        # X_train = list(self.embeddings) # copy
        # y_train = list(self.labels) # copy
        # x_test = X_train.pop(i)
        # y_test = y_train.pop(i)

        y_pred = get_knn_closest(self.embeddings, self.labels, [self.embeddings[i]], self.num_neighbours)
        print('YPRED: ',[self.labels[i]] + y_pred)
        return [self.labels[i]] + y_pred


def get_knn_closest(X_train, y_train, x_test, num_neighbours=5):
    # if type(x_test[0]) != list:
    #     x_test = list(x_test)
    knn = KNeighborsClassifier(num_neighbours+1)
    knn.fit(X_train, y_train)
    closest_inds = knn.kneighbors(x_test, return_distance=False)[0]
    neighbours = []
    for ind in closest_inds:
        neighbours.append(y_train[ind])
    return neighbours[1:]  # don't include x_test (is closest to itself)

if __name__ == '__main__':
    embeddings = [[1,1], [2,4], [5,4], [2,1], [1,1]]
    labels = ['a', 'b', 'b', 'a', 'a']
    eval(embeddings, labels)

