#!/bin/bash
script_path="ltu-0.0-lstm.py"

export LC_ALL=C.UTF-8

pip -q install pandas
pip -q install scikit-learn
pip -q install jupyterlab

cd notebooks/baselines
ipython $script_path