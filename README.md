[![Build Status](https://travis-ci.org/nelson-liu/paraphrase-id-tensorflow.svg?branch=master)](https://travis-ci.org/nelson-liu/paraphrase-id-tensorflow)
[![codecov](https://codecov.io/gh/nelson-liu/paraphrase-id-tensorflow/branch/master/graph/badge.svg)](https://codecov.io/gh/nelson-liu/paraphrase-id-tensorflow)

# paraphrase-id-tensorflow

Various models and code for paraphrase identification implemented in Tensorflow (1.1.0).

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
  
PR's to add more models / optimize or patch existing ones are more than welcome! The bulk of the model code resides in [duplicate_questions/models](https://github.com/nelson-liu/paraphrase-id-tensorflow/tree/master/duplicate_questions/models)

A lot of the data processing code is taken from / inspired by [allenai/deep_qa](https://github.com/allenai/deep_qa),
go check them out if you like how this project is structured!

## Installation

This project was developed in and has been tested on **Python 3.5** (it likely doesn't work with other versions of Python), 
and the package requirements are in [`requirements.txt`](./requirements.txt).

To install the requirements:

```
pip install -r requirements.txt
```

Note that after installing the requirements, you have to download the necessary NLTK
data by running (in your shell):

```
python -m nltk.downloader punkt
```

### GPU Training and Inference

Note that the [`requirements.txt`](./requirements.txt) file specify `tensorflow`
as a dependency, which is a CPU-bound version of tensorflow. If you have a GPU,
you should uninstall this CPU TensorFlow and install the GPU version by running:

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

Here's an example run command for the baseline Siamese BiLSTM:

```
python scripts/run_model/run_siamese.py train --share_encoder_weights --model_name=baseline_siamese --run_id=0
```

Here's an example run command for the Siamese BiLSTM with matching layer:

```
python scripts/run_model/run_siamese_matching_bilstm.py train --share_encoder_weights --model_name=siamese_matching --run_id=0
```


Here's an example run command for the BiMPM model:

```
python scripts/run_model/run_bimpm.py train --early_stopping_patience=5 --model_name=biMPM --run_id=0
```

Note that the defaults might not be ideal for your use, so feel free to turn the
knobs however you like.

## Contributors

- [Nelson Liu](http://nelsonliu.me)
- [Omar Khan](https://github.com/ohkhan)

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
