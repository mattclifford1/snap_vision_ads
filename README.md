# Snap Vision Data Pipeline
The [miro board](https://miro.com/app/board/uXjVOGzD1U0=/) contains a visual representation of the pipeline and ideas/ ongoing work
![Alt text](snap-vision-data-pipeline.png?raw=true "Data Pipeline")

# Running the Pipeline
To run the whole pipeline use:
```
$ python pipeline.py
```
To display the available command line arguments for the pipeline:

```
$ python pipeline.py --help
```

### Uselink links
 - [Unsupervised Deep Embedding for Clustering Analysis](https://arxiv.org/pdf/1511.06335.pdf)
 - [Kaggle triplet loss](https://www.kaggle.com/code/hirotaka0122/triplet-loss-with-pytorch/notebook)
 - [pytorch dataloader](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)


# Downloading the dataset
Follow the instructions inside [data](./data) to download and unzip the dataset automatically using python.

# Python Setup
## env with conda
```
$ conda create -n snap_vision python=3.8
$ conda activate snap_vision
```
## Install requirements
```
$ pip install -r requirements.txt
```
## Install pytorch CPU only
```
$ conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
## Or install on GPU (if you have nvidia GPU)
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
