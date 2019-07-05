#!/bin/bash
notebook_path="ltu-0.0-lstm.ipynb"
notebook_out_path="/artifacts/ltu-0.0-lstm.ipynb"

export LC_ALL=C.UTF-8

pip install papermill
pip install pandas
pip install scikit-learn

cd notebooks/baselines
papermill $notebook_path $notebook_out_path --log-output