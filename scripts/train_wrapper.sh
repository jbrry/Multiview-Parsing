#!/bin/bash

# sbatch options: | long partition: #SBATCH -p long | 24GB GPU: #SBATCH --gres=gpu:rtx6000:1 | 12GB GPU: #SBATCH --gres=gpu:rtx2080ti:1

test -z $1 && echo "Missing list of TBIDs (space or colon-separated)"
test -z $1 && exit 1
TBIDS=$(echo $1 | tr ':' ' ')

test -z $2 && echo "Missing model type: `dependency_parser`, `meta_parser`"
test -z $2 && exit 1
MODEL_TYPE=$2

test -z $3 && echo "Missing feature type, e.g.: `word`, `char`, `word_char`, `bert`, `word_char_meta`, `word_sentence-char_token-char_bert`"
test -z $3 && exit 1
FEATURE_TYPE=$3

# path to PyTorch checkpoints of model
MODEL_DIR="../ga_BERT/Irish-BERT/models/ga_bert/output/pytorch/electra/conll17_gdrive_NCI_oscar_paracrawl_filtering_basic+char-1.0+lang-0.8"
MODEL_PREFIX="electra-base-irish-cased-discriminator"

STEPS=(
   "100000"
   "200000"
   "300000"
   "400000"
   "500000"
   "600000"
   "700000"
   "800000"
   "900000"
   "1000000"
)

for STEP_SIZE in "${STEPS[@]}"; do
    echo "Running model with checkpoint: $STEP_SIZE"
    MODEL="${MODEL_PREFIX}-${STEP_SIZE}"
    MODEL_NAME="$MODEL_DIR/$MODEL"

    if [ ! -f "$MODEL_NAME/weights.tar.gz" ]; then
        echo "$MODEL_NAME/weights.tar.gz not found"
        exit 1
    fi

    ./scripts/train_dependency_parser.sh $TBIDS $MODEL_TYPE $FEATURE_TYPE $MODEL_NAME $STEP_SIZE

done
