# Snap Vision Data Pipeline
The [miro board](https://miro.com/app/board/uXjVOGzD1U0=/) contains a visual representation of the pipeline and ideas/ ongoing work
![Alt text](misc/snap-vision-data-pipeline.png?raw=true "Data Pipeline")

# Running the Pipeline
There are two pipelines to run, training and evaluation, make sure you have python set up with the correct packages installed before running by following the section below. If it is the first time running the pipeline it may take a little longer to run as the dataset download and rescaling will need to be done, but these are both stored locally inside [data](data), ready for future pipeline runs.
## Training
To run the pipeline in it's most basic form use:
```
$ python pipeline.py
```
There are many different options to train, such as what model type, where to save the trained models and various neural network training hyper parameters (if training neural network model type). You can display all of the available command line arguments for the pipeline by using the command:
```
$ python pipeline.py --help
```
### Neural Network Training
To train a neural network it is advised to use a GPU (the default batch sizes are for a 24GB VRAM card so adjust accordingly to your hardware). The available network currently are: [simple_net](models/toy_network.py), [big_net](models/network.py) and [facenet](models/FaceNet.py). To train facenet for 100 epochs use:
```
$ python pipeline.py --models_list facenet --epochs 100 --batch_size 64 --learning_rate 0.0001 --save_dir data/files_to_gitignore/models
```
The model weights will be saved inside the `save_dir` along with training stats details. If a model doesn't finish training or you would like to train a model for longer after training then increase the epochs. Any previous model weights found which have identical training hyperparamters will be automatically loaded and training will continue where it was left. So make sure you delete any previous models if you want to re train from scratch!

## Evaluation
To run the evaluation pipeline to display the closest embeddings use:
```
$ python evaluate.py
```
You can specify different models with the `--model` command line argument for example:
```
$ python evaluate.py --model simple_net
```
Will evaluate the [simple_net](models/toy_network.py) model with the latest training checkpoint.
### Example Closest embeddings from simple_net
![Alt text](misc/eg1.png?raw=true "Data Pipeline")
![Alt text](misc/eg2.png?raw=true "Data Pipeline")
![Alt text](misc/eg3.png?raw=true "Data Pipeline")
![Alt text](misc/eg4.png?raw=true "Data Pipeline")
![Alt text](misc/eg5.png?raw=true "Data Pipeline")
![Alt text](misc/eg6.png?raw=true "Data Pipeline")
![Alt text](misc/eg7.png?raw=true "Data Pipeline")
![Alt text](misc/eg8.png?raw=true "Data Pipeline")
![Alt text](misc/eg9.png?raw=true "Data Pipeline")
![Alt text](misc/eg10.png?raw=true "Data Pipeline")
![Alt text](misc/eg11.png?raw=true "Data Pipeline")
![Alt text](misc/eg12.png?raw=true "Data Pipeline")
### Extra options
By default all images from the eval set are show, to just show cases where the closest embeddings contain a correct similar image use:
```
$ python evaluate.py --show_case pass
```
or only where the closest embeddings dont contain a similar image:
```
$ python evaluate.py --show_case fail
```
By default only one image with its closest embeddings is shown, to increase it use:
```
$ python evaluate.py --num_disp 10
```
To save all eval examples to an image file at `data/files_to_gitignore/eval_figs` use:
```
$ python evaluate.py --num_disp 100 --save_fig
```
By default 5 of the closest embeddings are shown, to change it use:
```
$ python evaluate.py --num_neighbours 7
```
### View all evaluations
images of all embeddings from simple and simple_net model can be found [evaluation/eval_figs](evaluation/eval_figs)
### Download model weights from the cloud
If you don't have the model's weights stored locally, the download them from the cloud using:
```
$ python evaluate.py --model simple_net --download_weights
```
Which downloads the default simple neural network model. Use the `---weights_url` argument to change to a different model's storage location on the cloud. Model's must come in a zipped format.

### Load earlier checkpoints
Use the `--checkpoint` argument with the number at which epoch the weights are saved at if you wish to evaluate previous models.



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
# Downloading the dataset
The pipeline will automatically download and unzip the dataset, as well as downscale the image resolution, but if you want to just download the dataset in isolation then follow the instructions inside [data](./data) to download and unzip the dataset automatically using python.

# Uselink links
- [Unsupervised Deep Embedding for Clustering Analysis](https://arxiv.org/pdf/1511.06335.pdf)
- [Kaggle triplet loss](https://www.kaggle.com/code/hirotaka0122/triplet-loss-with-pytorch/notebook)
- [pytorch dataloader](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
