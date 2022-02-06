#!/bin/bash

test -z $1 && echo "Missing list of TBIDs (space or colon-separated)"
test -z $1 && exit 1
TBIDS=$(echo $1 | tr ':' ' ')

LOGDIR=$2
MODEL=$3
OUTDIR=$4

mkdir -p $OUTDIR

TB_DIR=data/ud-treebanks-v2.8

for tbid in $TBIDS; do
	echo "** predicting $tbid **"

	for filepath in ${TB_DIR}/*/${tbid}-ud-train.conllu; do
		dir=`dirname $filepath`        # e.g. /home/user/ud-treebanks-v2.2/UD_Afrikaans-AfriBooms
		tb_name=`basename $dir`        # e.g. UD_Afrikaans-AfriBooms

		# ud v2.x
		TRAIN_DATA_PATH=${TB_DIR}/${tb_name}/${tbid}-ud-train.conllu
		DEV_DATA_PATH=${TB_DIR}/${tb_name}/${tbid}-ud-dev.conllu
		TEST_DATA_PATH=${TB_DIR}/${tb_name}/${tbid}-ud-test.conllu

		for ftype in dev test; do
			if [[ "$ftype" == "dev" ]]; then
				TARGET=$DEV_DATA_PATH
			elif [[ "$ftype" == "test"  ]]; then
				TARGET=$TEST_DATA_PATH
			fi

			# if [[ -f "${LOGDIR}-${tbid}/model.tar.gz" ]]; then
    		# 		echo "found model ${LOGDIR}-${tbid}/model.tar.gz"
			# 	else 
    		# 		echo "${LOGDIR}-${tbid}/model.tar.gz does not exist."
			# 		exit 1
			# 	fi
			
			#MODEL_DIR=${LOGDIR}-${tbid}
			OUTFILE=${OUTDIR}/${tbid}-ud-${ftype}-${MODEL}.conllu
			EVAL_FILE=${OUTDIR}/${tbid}-ud-${ftype}-${MODEL}-eval.txt

			allennlp predict ${LOGDIR}/model.tar.gz $TARGET \
				--output-file ${OUTFILE} \
				--predictor conllu-multitask-predictor \
				--predictor-args '{"head_name": "dependencies"}' \
				--include-package meta_parser \
				--use-dataset-reader \
				--multitask-head "TBID" \
				--batch-size 32 \
				--silent
				
				echo "== $MODEL = $ftype =="
				python scripts/conll18_ud_eval.py -v ${TARGET} ${OUTFILE} > ${EVAL_FILE} 
		done
	done
done








