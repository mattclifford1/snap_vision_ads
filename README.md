# Snap Vision Data Pipeline
The [miro board](https://miro.com/app/board/uXjVOGzD1U0=/) contains a visual representation of the pipeline and ideas/ ongoing work

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
