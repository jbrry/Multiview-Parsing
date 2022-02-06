#!/bin/sh

test -z $1 && echo "Missing list of TBIDs (space or colon-separated)"
test -z $1 && exit 1
TBIDS=$(echo $1 | tr ':' ' ')

test -z $2 && echo "Missing model type: `dependency_parser`, `meta_parser`, `multitask`"
test -z $2 && exit 1
MODEL_TYPE=$2

test -z $3 && echo "Missing feature type, e.g.: `word`, `char`, `word_char`, `word_char_meta`, `word_sentence-char_token-char_bert`, `transformer`"
test -z $3 && exit 1
FEATURE_TYPE=$3

test -z $4 && echo "Missing model name, e.g.: path to local pytorch model or huggingface model name."
test -z $4 && exit 1
MODEL_NAME=$4

# optional metadata
METADATA=$5

TB_DIR=data/ud-treebanks-v2.8

TIMESTAMP=`date "+%Y%m%d-%H%M%S"`

for tbid in $TBIDS ; do
  # repeat experiments with different random_seed, numpy_seed and pytorch_seed
  #for RANDOM_SEED in 57682 86432 13337 91487 77663; do
  for RANDOM_SEED in 13337; do
    NUMPY_SEED=`echo $RANDOM_SEED | cut -c1-4`
    PYTORCH_SEED=`echo $RANDOM_SEED | cut -c1-3`

    export RANDOM_SEED=${RANDOM_SEED}
    export NUMPY_SEED=${NUMPY_SEED}
    export PYTORCH_SEED=${PYTORCH_SEED}

    export MODEL_NAME=${MODEL_NAME}
    echo "using ${MODEL_NAME}"
    
    echo
    echo "Now traning on $tbid"
    echo "-----------"


    CONFIG=configs/${MODEL_TYPE}/dependency_parser_source_transformer.jsonnet

    python run_metaparser.py --config $CONFIG \
    --tbids fr_sequoia \
    --model-type source \
    --pretrained-model-name $MODEL_NAME \
    --multi-transformer \

  done
done

# Baseline 1: XLM-R as source
#python run_metaparser.py --config dependency_parser_source_transformer.jsonnet --tbids ga_idt --model-type source
#python run_metaparser.py --config configs/meta_parser/dependency_parser_source_xlmr.jsonnet --tbids en_lines --model-type source
#python run_metaparser.py --config configs/meta_parser/dependency_parser_source_xlmr.jsonnet --tbids da_ddt --model-type source


# Baseline 2: Monolingual LM as source
#python run_metaparser.py --config configs/meta_parser/dependency_parser_source_transformer.jsonnet --tbids ga_idt --model-type source --pretrained-model-name DCU-NLP/bert-base-irish-cased-v1

#python run_metaparser.py --config configs/meta_parser/dependency_parser_source_transformer.jsonnet --tbids en_lines --model-type source --pretrained-model-name bert-base-cased

#python run_metaparser.py --config configs/meta_parser/dependency_parser_source_transformer.jsonnet --tbids da_ddt --model-type source --pretrained-model-name flax-community/roberta-base-danish


# ==== Mono: word
#python run_metaparser.py --config configs/meta_parser/dependency_parser_source_transformer-target_word.jsonnet --tbids ga_idt en_lines da_ddt --model-type meta

# separate optimizer
#python run_metaparser.py --config configs/meta_parser/dependency_parser_source_transformer-target_word_so.jsonnet --tbids ga_idt en_lines da_ddt --model-type meta

# ==== Mono: token-char
#python run_metaparser.py --config configs/meta_parser/dependency_parser_source_transformer-target_token-char.jsonnet --tbids ga_idt en_lines da_ddt --model-type meta --multi-transformer --metadata cross-stitch

#python run_metaparser.py --config configs/meta_parser/dependency_parser_source_transformer-target_token-char_so.jsonnet --tbids ga_idt en_lines da_ddt --model-type meta

# ==== Mono: sentence-char
#python run_metaparser.py --config configs/meta_parser/dependency_parser_source_transformer-target_sentence-char.jsonnet --tbids ga_idt en_lines da_ddt --model-type meta

#python run_metaparser.py --config configs/meta_parser/dependency_parser_source_transformer-target_sentence-char_so.jsonnet --tbids ga_idt en_lines da_ddt --model-type meta

# ==== Mono: word+token-char+sentence-char
#python run_metaparser.py --config configs/meta_parser/dependency_parser_source_transformer-target_word_token-char_sentence-char.jsonnet --tbids ga_idt en_lines da_ddt --model-type meta

# ==== Mono: deck of monolingual XLM-R
#python run_metaparser.py --config configs/meta_parser/dependency_parser_source_transformer-target_transformer.jsonnet --tbids ga_idt en_lines da_ddt --model-type meta --multi-transformer --mono-transformer

#python run_metaparser.py --config configs/meta_parser/dependency_parser_source_transformer-target_transformer.jsonnet --tbids fr_sequoia --model-type meta --multi-transformer --mono-transformer --use-cross-stitch --metadata cross-stitch

