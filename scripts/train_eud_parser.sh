#!/bin/bash

# sbatch options: | long partition: #SBATCH -p long | 24GB GPU: #SBATCH --gres=gpu:rtx6000:1 | 12GB GPU: #SBATCH --gres=gpu:rtx2080ti:1
# ./scripts/train_eud_parser.sh fr_sequoia enhanced transformer xlm-roberta-base enhanced_kg_parser enhanced_dependencies

test -z $1 && echo "Missing list of TBIDs (space or colon-separated)"
test -z $1 && exit 1
TBIDS=$(echo $1 | tr ':' ' ')

test -z $2 && echo "Missing  model type: `enhanced`"
test -z $2 && exit 1
MODEL_TYPE=$2

test -z $3 && echo "Missing feature type, e.g.: `word`, `char`, `word_char`, `word_char_meta`, `transformer`"
test -z $3 && exit 1
FEATURE_TYPE=$3

test -z $4 && echo "Missing model name, e.g.: path to local pytorch model or huggingface model name e.g. `xlm-roberta-base`."
test -z $4 && exit 1
MODEL_NAME=$4

test -z $5 && echo "Missing edge model type: `enhanced_dm_parser`, `enhanced_kg_parser`"
test -z $5 && exit 1
EDGE_MODEL_TYPE=$5

# heads
test -z $6 && echo "Missing list of heads: `enhanced_dependencies`"
test -z $6 && exit 1
HEADS=$6

# run ./scripts/download_iwpt_data.sh
TB_DIR=data/train-dev/

TIMESTAMP=`date "+%Y%m%d-%H%M%S"`

for TBID in $TBIDS ; do
  # repeat experiments with different random_seed, numpy_seed and pytorch_seed
  for RANDOM_SEED in 57682 86432 13337 91487 77663; do

    NUMPY_SEED=`echo $RANDOM_SEED | cut -c1-4`
    PYTORCH_SEED=`echo $RANDOM_SEED | cut -c1-3`

    export RANDOM_SEED=${RANDOM_SEED}
    export NUMPY_SEED=${NUMPY_SEED}
    export PYTORCH_SEED=${PYTORCH_SEED}

    export MODEL_NAME=${MODEL_NAME}
    echo "using ${MODEL_NAME}"
    
    echo
    echo "Now traning on $TBID"
    echo "-----------"

    for filepath in ${TB_DIR}/*/${TBID}-ud-train.conllu; do
      dir=`dirname $filepath`        # e.g. /home/user/ud-treebanks-v2.2/UD_Afrikaans-AfriBooms
      TREEBANK=`basename $dir`        # e.g. UD_Afrikaans-AfriBooms

      # ud v2.x
      export TRAIN_DATA_PATH=${TB_DIR}/${TREEBANK}/${TBID}-ud-train.conllu
      export DEV_DATA_PATH=${TB_DIR}/${TREEBANK}/${TBID}-ud-dev.conllu
      export TEST_DATA_PATH=${TB_DIR}/${TREEBANK}/${TBID}-ud-test.conllu

      export TBID=${TBID}
      export TREEBANK=${TREEBANK}
      export EDGE_MODEL_TYPE=${EDGE_MODEL_TYPE}

      CONFIG=configs/${MODEL_TYPE}/eud_${FEATURE_TYPE}_backbone_enhanced_dependencies.jsonnet

      allennlp train $CONFIG \
       -s logs/iwpt2021/${TBID}-${FEATURE_TYPE}-${EDGE_MODEL_TYPE}-${HEADS}-${RANDOM_SEED}-${TIMESTAMP} \
       --include-package meta_parser

    done
  done

done
