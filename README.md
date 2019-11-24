# Deep Math

## What is it?
Implementation of DeepMind's _Analysing Mathematical Reasoning Abilities of Neural Models_.

Link to paper: [https://openreview.net/pdf?id=H1gR5iR5FX](https://openreview.net/pdf?id=H1gR5iR5FX)

Link to dataset repository: [https://github.com/deepmind/mathematics_dataset](https://github.com/deepmind/mathematics_dataset)


## Overview
DeepMind's paper introduces a framework to evaluate the mathematical reasoning abilities of sequence-to-sequence models across a diverse range of mathematical problems. The paper introduces a synthetic dataset of problems in areas such as algebra, calculus, etc, and comes with two types of tests:

* **Interpolation:** These are problems that resemble those seen during training and possess a similar degree of difficulty.
* **Extrapolation:** These are problems designed to test generalisation along different axes by including more numbers, compositions, or larger samplers. Models are expected to perform worse on the extrapolation set than the interpolation one.

We implement two baselines from the paper from scratch, namely the **simple LSTM** and the **attentional LSTM**. Due to limited
resources we train on the `arithmetic-*` subset for at most 3 epochs. We observe that the results tend towards the
published results and are therefore confident we could match the performance at 12 epochs (setting of published results).

## Training
We built a custom Keras data generator to encode the input and output texts on demand during training to save memory.
Additionally, we provide a set of `make` commands to download and pre-process the data into the needed form.

We used Paperspace's `Gradient° Notebooks` to run early experiments, and `Gradient°
Jobs` to automate the training of models for longer periods of time.

Finally, we evaluate the training performance on the two `interpolation` and `extrapolation`
test sets provided with the synthetic dataset. The reported metric is an exact match metric: 1 if every output characters match and
0 otherwise. This metric is implemented as a callback in TensorFlow such that it can be tracked by the Keras model during training.

### Simple LSTM
The simple LSTM consists of a single LSTM cell that is fed with the input sequence and its outputs are used to predict
the next character. The architecture is laid out in the following figure:

![LSTM](figures/simple-lstm.png "Simple LSTM architecture")

We integrated the "thinking steps" and used the same hyperparameters outlined in the paper.
### Attentional LSTM

This model is a seq2seq LSTM model therefore follows the encoder/decoder architecture. The whole input is fed through
encoder and at the end the state is passed to the decoder:

![A-LSTM](figures/attentional-lstm.png "Attentional LSTM architecture")

The attention mechanism weights the hidden states passed through the encoder to produce the outputs at each step in the following way:

![A](figures/attention-mechanism.png "Attentional mechanism")

For a detailed explanation of attentional seq2seq models checkout [this awesome blog post](http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/).

## Results

### Simple LSTM

Investigating the performance of the simple LSTM for all arithmetic sub-tasks reveals that the model displays gains in performance at each epoch. This holds for both the interpolation and the extrapolation set.

|          Interpolation set          |         Extrapolation set          |
| :---------------------------------: | :--------------------------------: |
| ![](figures/simple-lstm-interp.gif) | ![](figures/simple-lstm-extra.gif) |

### Attentional LSTM

*Coming soon ...*

## Getting started
Clone the repository and create a virtual environment

```bash
virtualenv --python=python3 env
```

Spin up the virtual environment and install the required packages:

```bash
source ./env/bin/activate
pip install -r requirements-{cpu or gpu}.txt
```

## Make commands
We provide several `make` commands which can be viewed with

```bash
make help
```

## Configuration
* When running `make sequence_data` the choice of math module and difficulty level is configured by `settings.json`. Data is stored in `data/processed/` as `math-module_dificulty-level.pkl`.

* To submit a Gradient job, login to Paperspace, create an API key and add it to a credentials file with the following profile:

    **~/.paperspace/credentials**

    ```
    [tensor-league]
    api_token=AKIAIOSFODNN7EXAMPLE
    ```
