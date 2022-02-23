#!/bin/bash

# sbatch options: | long partition: #SBATCH -p long | 24GB GPU: #SBATCH --gres=gpu:rtx6000:1 | 12GB GPU: #SBATCH --gres=gpu:rtx2080ti:1

test -z $1 && echo "Missing list of TBIDs (space or colon-separated)"
test -z $1 && exit 1
TBID=$1

test -z $2 && echo "Needs model type, singleview, singleview-concat"
test -z $2 && exit 1
MODEL_TYPE=$2

python run_multiview_parser.py --tbids $TBID --config configs/multiview/dependency_parser_singleview.jsonnet --model-type ${MODEL_TYPE} --head-type dependencies

