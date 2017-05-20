[![Build Status](https://travis-ci.com/nelson-liu/quora_duplicate_questions.svg?token=nHesW4GKxvzu1T87bps4&branch=master)](https://travis-ci.com/nelson-liu/quora_duplicate_questions)
[![codecov](https://codecov.io/gh/nelson-liu/quora_duplicate_questions/branch/master/graph/badge.svg?token=WdeMLb9nuw)](https://codecov.io/gh/nelson-liu/quora_duplicate_questions)

# paraphrase-id-tensorflow

Various models and code for paraphrase identification, natural language inference,
and semantic textual similarity tasks, implemented in Tensorflow (1.1.0).

I took great care to document the code and explain what I'm doing at various
steps throughout the models; hopefully it'll be didactic example code for those
looking to get started with Tensorflow!

So far, this repo has implemented:

- A basic Siamese LSTM baseline, loosely based on the model
  in
  [Mueller, Jonas and Aditya Thyagarajan. "Siamese Recurrent Architectures for Learning Sentence Similarity." AAAI (2016).](https://www.semanticscholar.org/paper/Siamese-Recurrent-Architectures-for-Learning-Sente-Mueller-Thyagarajan/6812fb9ef1c2dad497684a9020d8292041a639ff)
  
- A Siamese LSTM model with an added "matching layer", as described
  in
  [Liu, Yang et al. "Learning Natural Language Inference using Bidirectional LSTM model and Inner-Attention." CoRR abs/1605.09090 (2016)](https://www.semanticscholar.org/paper/Learning-Natural-Language-Inference-using-Bidirect-Liu-Sun/f93a0a3e8a3e6001b4482430254595cf737697fa).

- The more-or-less state of the art Bilateral Multi-Perspective Matching model
  from
  [Wang, Zhiguo et al. "Bilateral Multi-Perspective Matching for Natural Language Sentences." CoRR abs/1702.03814 (2017)](https://www.semanticscholar.org/paper/Bilateral-Multi-Perspective-Matching-for-Natural-L-Wang-Hamza/b9d220520a5da7d302107aacfe875b8e2977fdbe).
  
  PR's to add more models / optimize or patch existing ones are more than welcome!

## Installation

This project has been tested on Python 3.5, and the package requirements are
in [`requirements.txt`](./requirements.txt).

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

To train a model or load + predict with a model, then run the scripts in
`scripts/run_model/` with `python <script_path>`. You can get additional
documentation about the parameters they take by running `python <script_path>
-h`


## Contributing

Do you have ideas on how to improve this repo? Have a feature request, bug
report, or patch? Feel free to open an issue or PR, as I'm happy to address
issues and look at pull requests.

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- Original immutable data (e.g. Quora Question Pairs).
    |
    ├── logs               <- Logs from training or prediction, including TF model summaries.
    │
    ├── models             <- Serialized models.
    |
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
    │   ├── data           <- Scripts to clean and split data
    │   │
    │   └── run_model      <- Scripts to train and predict with models.
    │
    └── tests              <- Directory with unit tests.
