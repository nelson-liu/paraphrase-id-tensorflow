[![Build Status](https://travis-ci.com/nelson-liu/quora_duplicate_questions.svg?token=nHesW4GKxvzu1T87bps4&branch=master)](https://travis-ci.com/nelson-liu/quora_duplicate_questions)
[![codecov](https://codecov.io/gh/nelson-liu/quora_duplicate_questions/branch/master/graph/badge.svg?token=WdeMLb9nuw)](https://codecov.io/gh/nelson-liu/quora_duplicate_questions)

# ppid_nli_sts

Various models and code for paraphrase identification, natural language inference,
and semantic textual similarity tasks.

## Installation

This project relies on Python 3.5, and the package requirements are in
[`requirements.txt`](./requirements.txt).

To install the requirements:

```
pip install -r requirements.txt
```

### GPU Training and Inference

Note that the [`requirements.txt`](./requirements.txt) file specify `tensorflow`
as a dependency, which is a CPU-bound version of tensorflow. If you have a gpu,
you should uninstall this CPU tensorflow and install the GPU version by running:

```
pip uninstall tensorflow
pip install tensorflow-gpu
```

## Getting / Processing The Data

To begin, run the following to generate the auxiliary directories for storing
data, trained models, and logs:

```
make aux_dirs
```

In addition, if you want to use pretrained GloVe vectors, run:

```
make glove
```

which will download pretrained Glove vectors to `data/external/`. Extract the
files in that same directory.

### Quora Question Pairs

To use the Quora Question Pairs data, download the dataset from
[Kaggle](https://www.kaggle.com/c/quora-question-pairs) (may require an
account). Place the downloaded zip archives in `data/raw/`, and extract the
files to that same directory.

Then, run:

```
make quora_data
```

to automatically clean and process the data with the scripts in
`scripts/data/quora`.

## Running models

To train a model or load + predict with a model, then run the scripts in `scripts/run_model/` with `python <script_path>`. You can get additional documentation about the parameters they take by running `python <script_path> -h`

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- Original immutable data (e.g. SNLI / Quora Question Pairs).
    |
    ├── logs               <- Logs from training or prediction.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    |
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── duplicate_questions<- Module with source code for models and data.
    │   ├── data           <- Methods and classes for manipulating data.
    │   │
    │   ├── models         <- Methods and classes for training models.
    │   │
    │   └── util           <- Various helper methods and classes for use in models.
    │
    ├── scripts            <- Scripts for generating the data
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   └── run_model      <- Scripts to train and predict with models.
    │
    └── tests              <- Directory with unit tests.
