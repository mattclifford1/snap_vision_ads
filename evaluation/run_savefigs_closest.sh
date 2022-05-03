#!/bin/bash
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
VENV=snap_vision
conda activate $VENV


python evaluate.py --save_fig --num_disp 310 --show_case fail --model simple
python evaluate.py --save_fig --num_disp 310 --show_case pass --model simple
python evaluate.py --save_fig --num_disp 310 --show_case fail --model simple_net
python evaluate.py --save_fig --num_disp 310 --show_case pass --model simple_net