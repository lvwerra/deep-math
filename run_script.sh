#!/bin/bash
file_name=train_attention.py
export LC_ALL=C.UTF-8

pip -q install pandas
pip -q install scikit-learn
pip -q install jupyterlab
pip -q install click

python src/models/$file_name --settings settings.json