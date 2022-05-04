#!/bin/bash
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
VENV=snap_vision
conda activate $VENV


# python evaluate_visually.py --save_fig --num_disp 310 --show_case fail --model simple
# python evaluate_visually.py --save_fig --num_disp 310 --show_case pass --model simple
python evaluate_visually.py --save_fig --num_disp 310 --show_case fail --model simple_net
python evaluate_visually.py --save_fig --num_disp 310 --show_case pass --model simple_net
# python evaluate_visually.py --save_fig --num_disp 310 --show_case fail --model facenet
# python evaluate_visually.py --save_fig --num_disp 310 --show_case pass --model facenet
# python evaluate_visually.py --save_fig --num_disp 310 --show_case fail --model big_net
# python evaluate_visually.py --save_fig --num_disp 310 --show_case pass --model big_net
