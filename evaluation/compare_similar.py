'''
compare to nearest embeddings and see if they are of the same class
'''
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm


class eval:
    def __init__(self, embeddings, labels, num_neighbours=3):
        self.embeddings = embeddings   # list
        self.labels = labels           # list
        self.num_neighbours = num_neighbours
        self.compute()

    def compute(self):
        scores = []
        print('Running evaluation')
        for i in tqdm(range(len(self.labels))):
            X_train = list(self.embeddings) # copy
            y_train = list(self.labels) # copy
            x_test = X_train.pop(i)
            y_test = y_train.pop(i)
            y_pred = self.get_knn_class(X_train, y_train, [x_test])
            if y_pred[0] == y_test:
                scores.append(1)
            else:
                scores.append(0)
        self.accuracy = (sum(scores)/len(scores))
        print('Accuracy: ', self.accuracy*100, '%')

    def get_knn_class(self, X_train, y_train, x_test):
        neigh = KNeighborsClassifier(self.num_neighbours)
        neigh.fit(X_train, y_train)
        return neigh.predict(x_test)


if __name__ == '__main__':
    embeddings = [[1,1], [2,4], [5,4], [2,1], [1,1]]
    labels = ['a', 'b', 'b', 'a', 'a']
    eval(embeddings, labels)
