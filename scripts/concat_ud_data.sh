#!/usr/bin/env bash

test -z $1 && echo "Missing list of tbids (space or colon-separated), e.g. 'en_ewt:en_gum"
test -z $1 && exit 1
TBIDS=$(echo $1 | tr ':' ' ')

OUTDIR="data/ud-treebanks-v2.8-concatenated"
DATASET_DIR="data/ud-treebanks-v2.8"

echo "Generating concatenated dataset..."

if [[ ! -d "${DATASET_DIR}-concatenated" ]]; then
    echo "Creating output directory..."
    mkdir -p "${DATASET_DIR}-concatenated"
fi

python scripts/concat_treebanks.py ${OUTDIR} --dataset_dir ${DATASET_DIR} --treebank_ids ${TBIDS}
