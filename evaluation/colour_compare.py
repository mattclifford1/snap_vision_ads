'''
compare to nearest embeddings and see if they are of the same class
'''
# import multiprocessing
import numpy as np
import scipy
from skimage import io, color,filters
from sklearn.neighbors import KNeighborsClassifier
import torch.multiprocessing as multiprocessing
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
        y_pred = get_closest_colours(self.embeddings, self.labels, [self.embeddings[i]], self.num_neighbours)
        return [self.labels[i]] + y_pred


def get_closest_colours(embeddings, labels, dom_cols, num_neighbours = 5):

    result = {'label':[],'distance':[]}

    for m in range(len(embeddings)):
        
        e = embeddings[m]
        n = len(dom_cols)
        A = np.zeros([n,n])
        print('Dominant Colours: ',dom_cols,'\n','-'*50,'\n','Embeddings: ',e)
        i = 0
        j = 0

        # for d in data['dom_colours_lab'][0]:
        for t in range(0,n):
            for c in range(0,n):
                A[t,c] = color.deltaE_cie76(e[c],dom_cols[0][t])

        inds_A = scipy.optimize.linear_sum_assignment(A)

        sum_dist = 0
        for n_A in range(0,len(inds_A[0])):
            i = inds_A[0][n_A]
            j = inds_A[1][n_A]
        
            sum_dist = sum_dist + A[i][j]

        result['distance'].append(sum_dist)
        result['label'].append(labels[m])
    nn = num_neighbours+1
    ind = np.argpartition(result['distance'], -nn)[-nn:]     
    ind[np.argsort(result['distance'][ind])]   

    return result['labels'][ind][1:]






if __name__ == '__main__':
    embeddings = [[1,1], [2,4], [5,4], [2,1], [1,1]]
    labels = ['a', 'b', 'b', 'a', 'a']
    eval(embeddings, labels)

