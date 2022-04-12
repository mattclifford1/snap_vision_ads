# Data downloader
to download dataset using python:
```
data_dir = download.get_if_doesnt_exist(ARGS.dataset_dir)
```
or from the command line:
```
$ python data/download.py
```
# Data downscaler
```
data_dir = resize_dataset.run(data_dir, 512)
```

# Large data file
Put any file that you dont want git to track inside the files_to_gitignore directory
