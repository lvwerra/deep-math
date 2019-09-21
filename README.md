# Deep Math

## What is it?
Implementation of DeepMind's _Analysing Mathematical Reasoning Abilities of Neural Models_.

Link to paper: [https://openreview.net/pdf?id=H1gR5iR5FX](https://openreview.net/pdf?id=H1gR5iR5FX)

Link to dataset repository: [https://github.com/deepmind/mathematics_dataset](https://github.com/deepmind/mathematics_dataset)

## Getting started
Clone the repository and create a virtual environment

```bash
virtualenv --python=python3 env
```

Spin up the virtual environment and intall the required packages:

```bash
source ./env/bin/activate
pip install -r requirements-{cpu or gpu}.txt
```

## Make commands

### Get mathematics dataset
Downloads the pre-generated data from DeepMind and extracts to `data/raw`:

```
make dataset
```

### Generate sequence data for math module
Processes raw question-answer pairs into form needed for training models:

```
make sequence_data
```
Choice of math module and difficulty level configured by `settings.json`. Data is stored in `data/processed/` as`math-module_dificulty-level.pkl`.

### Submit Gradient job
Login to Paperspace, create an API key and add it to a credentials file with the following profile:

**~/.paperspace/credentials**

```
[tensor-league]
api_token=AKIAIOSFODNN7EXAMPLE
```

From here you can submit Gradient jobs with

```
make gradient_job
```

which can be configured via `settings.json` and the global variables in the `Makefile`.

## Useful commands

To concatenate all files from one module run:
```
find . \( -path "./train*" -a -name "*arithmetic*" \) -exec cat "{}" \; > concat/train.csv
```