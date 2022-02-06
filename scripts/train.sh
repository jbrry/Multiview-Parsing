 #!/usr/bin/env bash

TB_DIR=data/ud-treebanks-v2.6

test -z $1 && echo "Missing list of TBIDs (space or colon-separated)"
test -z $1 && exit 1
TBIDS=$(echo $1 | tr ':' ' ')

TIMESTAMP=`date "+%Y%m%d-%H%M%S"` 

for tbid in $TBIDS ; do
  echo
  echo "== $tbid =="
  echo
  for filepath in ${TB_DIR}/*/${tbid}-ud-train.conllu; do
    dir=`dirname $filepath`        # e.g. /home/user/ud-treebanks-v2.2/UD_Afrikaans-AfriBooms
    tb_name=`basename $dir`        # e.g. UD_Afrikaans-AfriBooms

    # ud v2.x
    export TRAIN_DATA_PATH=${TB_DIR}/${tb_name}/${tbid}-ud-train.conllu
    export DEV_DATA_PATH=${TB_DIR}/${tb_name}/${tbid}-ud-dev.conllu
    export TEST_DATA_PATH=${TB_DIR}/${tb_name}/${tbid}-ud-test.conllu

    allennlp train configs/dependency_parser.jsonnet -s logs/${tbid}-${TIMESTAMP}
  done
done
