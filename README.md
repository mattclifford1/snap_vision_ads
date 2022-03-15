# Set up python env with conda
```
$ conda create -n snap_vision python=3.8
$ conda activate snap_vision
```
# Install requirements
```
$ pip install -r requirements.txt
```
# Install pytorch CPU only
```
$ conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
# Or install on GPU (if you have nvidia GPU)
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
