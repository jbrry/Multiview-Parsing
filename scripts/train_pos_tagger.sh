#!/bin/bash

test -z $1 && echo "Missing list of TBIDs (space or colon-separated)"
test -z $1 && exit 1
TBIDS=$(echo $1 | tr ':' ' ')

test -z $2 && echo "Missing model type: `ablations`, `pos_tagging`, `meta_tagging`"
test -z $2 && exit 1
MODEL_TYPE=$2

test -z $3 && echo "Missing feature type: `word`, `char`, `word_char`, `word_char_meta`"
test -z $3 && exit 1
FEATURE_TYPE=$3

TB_DIR=data/ud-treebanks-v2.6
EMB_DIR=$HOME/spinning-storage/$USER/embeddings/

TIMESTAMP=`date "+%Y%m%d-%H%M%S"`

for tbid in $TBIDS ; do
  # repeat experiments with different random_seed, numpy_seed and pytorch_seed
  for RANDOM_SEED in 57682 86432 13337 91487 77663; do
    NUMPY_SEED=`echo $RANDOM_SEED | cut -c1-4`
    PYTORCH_SEED=`echo $RANDOM_SEED | cut -c1-3`

    export RANDOM_SEED=${RANDOM_SEED}
    export NUMPY_SEED=${NUMPY_SEED}
    export PYTORCH_SEED=${PYTORCH_SEED}
    
    echo
    echo "Now traning on $tbid"
    echo "-----------"

    for filepath in ${TB_DIR}/*/${tbid}-ud-train.conllu; do
      dir=`dirname $filepath`        # e.g. /home/user/ud-treebanks-v2.2/UD_Afrikaans-AfriBooms
      tb_name=`basename $dir`        # e.g. UD_Afrikaans-AfriBooms

      # ud v2.x
      export TRAIN_DATA_PATH=${TB_DIR}/${tb_name}/${tbid}-ud-train.conllu
      export DEV_DATA_PATH=${TB_DIR}/${tb_name}/${tbid}-ud-dev.conllu
      export TEST_DATA_PATH=${TB_DIR}/${tb_name}/${tbid}-ud-test.conllu

      # pretrained embeddings
      iso=$(echo ${tbid} | awk -F "_" '{print $1}')
      treebank_lang=$(echo ${tb_name} | awk -F "_" '{print $2}')
      lang=$(echo ${treebank_lang} | awk -F "-" '{print $1}')
      echo "processing language: ${lang}"
      VECS_DIR=${EMB_DIR}/${lang}
      ZIPPED_VECS_FILE=$(ls ${VECS_DIR}/*.xz)

      if [ -f "$ZIPPED_VECS_FILE" ]; then
          echo "embedding file $ZIPPED_VECS_FILE exists. Unzipping."
          unxz $ZIPPED_VECS_FILE
          VECS_FILE=$(ls ${VECS_DIR}/*.vectors)
          # sometimes there are utf-8 encoding errors even though the file is utf-8
          iconv -f utf-8 -t utf-8 -c ${VECS_FILE} > ${VECS_DIR}/${iso}-cleaned.txt
          VECS_FILE=${VECS_DIR}/${iso}-cleaned.txt
      else 
          echo "$ZIPPED_VECS_FILE not found. Assuming this has already been de-compressed and cleaned."
          # see if the cleaned version is available, create it if not
          VECS_FILE=$(ls ${VECS_DIR}/${iso}-cleaned.txt)
              if [ ! -e "${VECS_FILE}" ]; then
                  iconv -f utf-8 -t utf-8 -c ${VECS_DIR}/${iso}.vectors > ${VECS_DIR}/${iso}-cleaned.txt
                  VECS_FILE=${VECS_DIR}/${iso}-cleaned.txt
              fi
      fi

      echo "using embeddings: ${VECS_FILE}"
      export VECS_PATH=${VECS_FILE}

      allennlp train configs/${MODEL_TYPE}/pos_tagger_${FEATURE_TYPE}.jsonnet \
        -s logs/xpos_tagging_embeddings_5seeds/${tbid}-${MODEL_TYPE}-${FEATURE_TYPE}-${RANDOM_SEED}-${TIMESTAMP} \
        --include-package meta_parser

    done
  done
done
