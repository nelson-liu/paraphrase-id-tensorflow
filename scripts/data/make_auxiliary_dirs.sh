# Note that this script should NOT be run directly 
# (e.g. with `./make_auxiliary_dirs.sh`),
# but rather run through the Makefile (with `make aux_dirs`)

echo "Creating auxiliary directories"
mkdir -p ./data/external
mkdir -p ./data/interim
mkdir -p ./data/processed
mkdir -p ./data/raw
mkdir -p ./logs/
mkdir -p ./models/
echo "Done creating auxiliary directories"
