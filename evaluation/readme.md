# Evaluation
To print out the accuracy of the correct nearest embeddings use:
```
from evaluation import compare_similar

embeddings = [[1,1], [2,4], [5,4], [2,1], [1,1]]
labels = ['a', 'b', 'b', 'a', 'a']
evaluation = compare_similar.eval(embeddings, labels)  # will print out accuracy
accuracy = evaluation.accuracy    # accessing accuracy data
```
