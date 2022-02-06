#!/bin/bash

# sbatch options: | long partition: #SBATCH -p long | 24GB GPU: #SBATCH --gres=gpu:rtx6000:1 | 12GB GPU: #SBATCH --gres=gpu:rtx2080ti:1

test -z $1 && echo "Missing list of TBIDs (space or colon-separated)"
test -z $1 && exit 1
TBIDS=$(echo $1 | tr ':' ' ')

test -z $2 && echo "Missing model type: `dependency_parser`, `meta_parser`, `multitask`"
test -z $2 && exit 1
MODEL_TYPE=$2

test -z $3 && echo "Missing feature type, e.g.: `word`, `char`, `word_char`, `word_char_meta`, `word_sentence-char_token-char_bert`"
test -z $3 && exit 1
FEATURE_TYPE=$3

test -z $4 && echo "Missing model name, e.g.: path to local pytorch model or huggingface model name."
test -z $4 && exit 1
MODEL_NAME=$4

# optional metadata
METADATA=$5

TB_DIR=data/ud-treebanks-v2.8

TIMESTAMP=`date "+%Y%m%d-%H%M%S"`

for TBID in $TBIDS ; do
  # repeat experiments with different random_seed, numpy_seed and pytorch_seed
  for RANDOM_SEED in 13337; do

    NUMPY_SEED=`echo $RANDOM_SEED | cut -c1-4`
    PYTORCH_SEED=`echo $RANDOM_SEED | cut -c1-3`

    export RANDOM_SEED=${RANDOM_SEED}
    export NUMPY_SEED=${NUMPY_SEED}
    export PYTORCH_SEED=${PYTORCH_SEED}

    export MODEL_NAME=${MODEL_NAME}
    echo "using ${MODEL_NAME}"
    export TBID=${TBID}
    
    echo
    echo "Now traning on $TBID"
    echo "-----------"

    for filepath in ${TB_DIR}/*/${TBID}-ud-train.conllu; do
      dir=`dirname $filepath`        # e.g. /home/user/ud-treebanks-v2.2/UD_Afrikaans-AfriBooms
      tb_name=`basename $dir`        # e.g. UD_Afrikaans-AfriBooms

      # ud v2.x
      export TRAIN_DATA_PATH=${TB_DIR}/${tb_name}/${TBID}-ud-train.conllu
      export DEV_DATA_PATH=${TB_DIR}/${tb_name}/${TBID}-ud-dev.conllu
      export TEST_DATA_PATH=${TB_DIR}/${tb_name}/${TBID}-ud-test.conllu

      CONFIG=configs/${MODEL_TYPE}/tagger_parser_small.jsonnet
      LOGDIR=${TBID}-${MODEL_TYPE}-${FEATURE_TYPE}-${METADATA}-${RANDOM_SEED}-${TIMESTAMP}

      allennlp train $CONFIG \
        -s logs/dependency_parsing_gabert/${LOGDIR} \
        --include-package meta_parser

    done
  done
done
