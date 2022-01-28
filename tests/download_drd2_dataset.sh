#!/bin/bash

# This script will create a preprocessed dataset for the graph-to-graph translation of DRD2.
# If you want to run the DESMILES overview notebook,
# please place this dataset to ${DESMILES_DATA_DIR}/notebooks/

# Keep track of desired install path; move the dataset here in the end.
if [[ -d "$1" ]]; then
    data_path=$1
    echo "Downloading to ${data_path}"
else
    echo "Downloading to local directory"
    data_path=$(pwd)
fi

# Get the path for the utility scripts
utils=$(readlink -f $(dirname $0))/utils

# Create a temporary directory structure for the downloads and preprocessing
dir=$(mktemp -d)

drd2_dir=${dir}/DRD2
val_dir=${drd2_dir}/Validation
test_dir=${drd2_dir}/Testing

mkdir -p ${drd2_dir}
mkdir -p ${val_dir}
mkdir -p ${test_dir}


# We are using the specific repo of the paper published by Wengong Jin to download the data
DRD2_repo="https://raw.githubusercontent.com/wengong-jin/iclr19-graph2graph/691e28c12d9753c53b765932100d667885376d34"

cd ${test_dir}
curl -f -O ${DRD2_repo}/data/drd2/test.txt

cd ${val_dir}
curl -f -O ${DRD2_repo}/data/drd2/valid.txt

cd ${drd2_dir}
curl -f -O ${DRD2_repo}/data/drd2/train_pairs.txt

# Perform a series of preprocessing steps for DESMILES 
${utils}/get_smiles_from_pairs.sh train_pairs.txt > drd2.smi
${utils}/apply_bpe.sh drd2.smi drd2.enc8000
${utils}/convert_to_np.py drd2.enc8000 drd2.enc8000.npy
${utils}/get_fingerprints.py drd2.smi fps_drd2.npz

# Now download the pickled scoring function and use it to evaluate the scores:
curl -f -O ${DRD2_repo}/props/clf_py36.pkl
${utils}/compute_drd2_probs.py drd2.smi drd2_probs.npy
${utils}/convert_to_canon_smiles.py drd2.smi drd2_canon_smiles.smi

# Move the processed DRD2 data directory to the current location.
mv ${drd2_dir} ${data_path}

# Clean up the tmp directory
rm -rf ${dir}
