# data loading
to load and access the database
```
from data_loader.load import get_database
data = get_database()
instance_0 = data[0]
```
looping over the database rows:
```
for row in data:
    im_path = row['image_path']
    similar_path = row['similar_paths']
```
