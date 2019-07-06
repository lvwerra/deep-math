#!/bin/bash
notebook_path="ltu-0.0-lstm.ipynb"
notebook_out_path="/artifacts/$notebook_path"

export LC_ALL=C.UTF-8

pip -q install papermill
pip -q install pandas
pip -q install scikit-learn
pip -q install gradient_statsd

cd notebooks/baselines
papermill $notebook_path $notebook_out_path --log-output