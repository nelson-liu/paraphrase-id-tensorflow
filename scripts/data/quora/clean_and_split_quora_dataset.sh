#!/usr/bin/env bash
# Create the cleaned, split final quora dataset from the raw text files
# The raw text files are assumed to be put in "./data/processed/raw"

# cd to the path of this bash file
parent_path=$( cd "$(dirname "${BASH_SOURCE}")" ; pwd -P )

cd "$parent_path"

echo "Making quora folder in data/interim/ if it does not exist"
# Make a folder in the interim/ data folder to hold the interim quora data
mkdir -p ../../../data/interim/quora/

echo "Cleaning the raw train and test CSVs"
# Clean the raw train csv and test csv file
python3 clean_quora_dataset.py ../../../data/raw/train.csv ../../../data/interim/quora/
python3 clean_quora_dataset.py ../../../data/raw/test.csv ../../../data/interim/quora/

echo "Making quora folder in data/processed/ if it does not exist"
# Make a folder in the processed/ data folder to hold the processed quora data
mkdir -p ../../../data/processed/quora/

echo "Splitting the train CSV into train and val and moving it to data/processed/quora"
# Split the cleaned train dataset into train and validation splits, and
# write the output to the processed/ data folder.
python3 split_quora_file.py 0.1 ../../../data/interim/quora/train_cleaned.csv ../../../data/processed/quora/

echo "Copying the cleaned test CSV into data/processed/quora"
# Copy the cleaned test file to the processed/ data folder.
cp ../../../data/interim/quora/test_cleaned.csv ../../../data/processed/quora/test_final.csv
