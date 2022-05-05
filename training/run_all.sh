#!/bin/bash
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
VENV=snap_vision
conda activate $VENV

python pipeline.py --models_list simple_net --learning_rate 0.001  --lr_decay 0.98
python pipeline.py --models_list simple_net --learning_rate 0.01   --lr_decay 0.98
python pipeline.py --models_list simple_net --learning_rate 0.0001 --lr_decay 0.98
