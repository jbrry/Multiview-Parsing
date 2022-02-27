#!/bin/bash

# sbatch options: | long partition: #SBATCH -p long | 24GB GPU: #SBATCH --gres=gpu:rtx6000:1 | 12GB GPU: #SBATCH --gres=gpu:rtx2080ti:1

test -z $1 && echo "Missing list of TBIDs (space or colon-separated)"
test -z $1 && exit 1
TBIDS=$(echo $1 | tr ':' ' ')

test -z $2 && echo "Needs model type, singleview, singleview-concat, multiview"
test -z $2 && exit 1
MODEL_TYPE=$2


for TBID in $TBIDS ; do
    echo "running $TBID"
    sbatch -J ${TBID} --gres=gpu:rtx2080ti:1 ./scripts/train_worker_singleview.sh ${TBID} ${MODEL_TYPE}
    sleep 10
done


