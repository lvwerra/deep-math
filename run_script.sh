#!/bin/bash
file_name=train_lstm.py
export LC_ALL=C.UTF-8

pip -q install pandas
pip -q install scikit-learn
pip -q install jupyterlab

python src/models/$file_name