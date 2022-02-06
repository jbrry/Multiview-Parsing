#!/bin/bash

cd data

echo "Downloading UD data..."$'\n'

curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3687{/ud-treebanks-v2.8.tgz}


tar -xvzf ud-treebanks-v2.8.tgz
rm ud-treebanks-v2.8.tgz
echo $'\n'"Done"
