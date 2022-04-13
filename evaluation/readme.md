# Evaluation
To print out the accuracy of the correct nearest embeddings use:
```
from evaluation import nearest_points

embeddings = [[1,1], [2,4], [5,4], [2,1], [1,1]]
labels = ['a', 'b', 'b', 'a', 'a']
evaluation = nearest_points.eval(embeddings, labels)  # will print out accuracy
accuracy = evaluation.accuracy    # accessing accuracy data
```

Evalution is done by taking the nearest embeddings of a data point's embedding and assigning it the same class.
Evaluation is sped up by using the `multiprocessing` python library to compute whether the closest embeddings are correct or not in parallel.
