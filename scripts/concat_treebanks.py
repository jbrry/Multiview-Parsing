# adapted from: https://github.com/Hyperparticle/udify/blob/master/concat_treebanks.py

"""
Concatenates treebanks together
"""

from typing import List, Tuple, Dict, Any

import os
import re
import glob
import shutil
import logging
import argparse


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("--output-dir", type=str, help="The path to output the concatenated files")
parser.add_argument("--dataset-dir", default="data/ud-treebanks-v2.8", type=str,
                    help="The path containing all UD treebanks")
parser.add_argument("--treebank-ids", default=[], type=str, nargs="+",
                    help="Specify a list of treebank IDs to use")

args = parser.parse_args()

def get_ud_treebank_files(dataset_dir: str, treebank_ids: List[str] = None) -> Dict[str, Tuple[str, str, str]]:
    """
    Retrieves all treebank data paths in the given directory.
    :param dataset_dir: the directory where all treebank directories are stored
    :param treebank_ids: retrieve just the subset of treebank IDs listed here
    :return: a dictionary mapping a treebank name to a list of train, dev, and test conllu files
    """
    datasets = {}
    
    for tbid in treebank_ids:
        train_file = tbid + "-ud-train.conllu"
        pathname = os.path.join(dataset_dir, "*", train_file)
        train_path = glob.glob(pathname).pop()
        treebank_path = os.path.dirname(train_path)
        treebank = os.path.basename(treebank_path)
        datasets[treebank] = {}

        train_file = [file for file in os.listdir(treebank_path) if file.endswith("train.conllu")]
        dev_file = [file for file in os.listdir(treebank_path) if file.endswith("dev.conllu")]
        test_file = [file for file in os.listdir(treebank_path) if file.endswith("test.conllu")]

        train_file = os.path.join(treebank_path, train_file[0]) if train_file else None
        dev_file = os.path.join(treebank_path, dev_file[0]) if dev_file else None
        test_file = os.path.join(treebank_path, test_file[0]) if test_file else None

        datasets[treebank]["train"] = train_file
        datasets[treebank]["dev"] = dev_file
        datasets[treebank]["test"] = test_file

    return datasets


datasets = get_ud_treebank_files(args.dataset_dir, args.treebank_ids)

all_treebank_files = [datasets[t] for t in datasets.keys()]
all_treebanks = "+".join(t for t in datasets.keys())

output_path = os.path.join(args.output_dir, all_treebanks)
if not os.path.exists(output_path):
    print(f"Creating output path {output_path}")
    os.makedirs(output_path)

basenames = [] # a None here will get rid of dev
tbids = [] # use a list to maintain order of tbids
merged_basenames = []

# for treebank, files in datasets.items():

for treebank, files in datasets.items():
    for file_path in files.values():
        try:
            basename = os.path.basename(file_path)
            tbid = basename.split("-")[0]
            if tbid not in tbids:
                tbids.append(tbid)
            basenames.append(basename)
        except TypeError:
            continue

all_tbids = "+".join(tbid for tbid in tbids)

for b in basenames:
    tbid = b.split("-")[0]
    merged = re.sub(r'\b' + tbid + r'\b', all_tbids, b)
    if merged not in merged_basenames:
        merged_basenames.append(merged)

for key in ["train", "dev", "test"]:
    file_group = []
    # collect all treebank files for the given split
    for treebank in datasets.keys():
        _file = datasets[treebank][key]
        file_group.append(_file)
    for i, b in enumerate(merged_basenames):
        if key in b:
            name = merged_basenames[i]
    
    # write
    assert key in name # just to be sure we're not mixing file types
    file_output_path = os.path.join(output_path, name)
    print()
    print(f"Writing file group {key} containing files: {file_group} to {file_output_path}")
    with open(file_output_path, 'w') as write:
        for _file in file_group:
            if not _file:
                continue
            with open(_file, 'r') as read:
                shutil.copyfileobj(read, write)

print("done")