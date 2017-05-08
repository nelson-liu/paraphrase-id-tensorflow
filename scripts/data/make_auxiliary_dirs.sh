# cd to the path of this bash file
parent_path=$( cd "$(dirname "${BASH_SOURCE}")" ; pwd -P )

cd "$parent_path"

echo "Creating auxiliary directories"
mkdir -p ../../data/external
mkdir -p ../../data/interim
mkdir -p ../../data/processed
mkdir -p ../../data/raw
mkdir -p ../../logs/
mkdir -p ../../models/
echo "Done creating auxiliary directories"
