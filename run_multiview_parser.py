"""
Script to launch the meta parser.
"""

import os
import sys
import json
import subprocess
import _jsonnet
import copy
import datetime
import logging
import argparse
import glob


parser = argparse.ArgumentParser()
parser.add_argument("--name", default="", type=str, help="Log dir name")
parser.add_argument(
    "--config",
    default="configs/multiview/dependency_parser_multiview.jsonnet",
    type=str,
    help="Configuration file",
)
parser.add_argument(
    "--dataset_dir",
    default="data/ud-treebanks-v2.8",
    type=str,
    help="The path containing all UD treebanks",
)
parser.add_argument(
    "--tbids", default=[], type=str, nargs="+", help="Treebank ids to include"
)
parser.add_argument(
    "--model-type",
    choices=["singleview", "multiview"],
    type=str,
    help="Model type: singleview is a single parsing head,"
    " multiview has an additional encoder which passes its outputs to the meta view.",
)
parser.add_argument(
    "--head-type",
    choices=["dependencies"],
    type=str,
    help="Head type: so far we just consider dependencies, but this could be POS etc."
)
parser.add_argument(
    "--do-lower-case",
    default=False,
    action="store_true",
    help="Whether to do lower case or not"
)
parser.add_argument(
    "--use-cross-stitch",
    default=False,
    action="store_true",
    help="Whether to use a cross-stitch layer in the meta model."
)
parser.add_argument(
    "--pretrained-model-name",
    type=str,
    default="xlm-roberta-base",
    help="Name of pretrained transformer model",
)
parser.add_argument(
    "--metadata",
    type=str,
    help="Metadata to inclde to experiment run.",
)
parser.add_argument(
    "--random-seed",
    type=str,
    default="13337",
    help="Initial random seed.",
)
parser.add_argument(
    "--fields",
    default=[
        "dataset_reader",
        "model",
        "heads",
        "train_data_path",
        "validation_data_path",
        "test_data_path",
        "validation_metric",
        "random_seed",
        "numpy_seed",
        "pytorch_seed",
    ],
    type=str,
    nargs="+",
    help="Config fields to modify",
)

TBID_PLACEHOLDER="TBID_PLACEHOLDER"

args = parser.parse_args()

jsonnet_file = args.config
data = json.loads(_jsonnet.evaluate_file(jsonnet_file))

basename = os.path.basename(args.config)
dirname = os.path.dirname(args.config)

dest = dirname.split("/")
dest.append("tmp")
dest_dir = "/".join(dest)

original_json_file = os.path.splitext(basename)

model_identifier = args.pretrained_model_name.split("/")[-1]
if args.metadata:
    metadata_string = f"-{args.metadata}"
else:
    metadata_string = ""

run_name = original_json_file[0] + f"-{model_identifier}-" + "-".join(args.tbids) + metadata_string + original_json_file[1]

logdir = f"logs/dependency_parsing_multiview/{run_name}"

dest.append(run_name)
dest_file = "/".join(dest)

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

nb_tbids = len(args.tbids)

print(f"creating config {run_name}")
for i, tbid in enumerate(args.tbids, start=1):
    print(f"=== {tbid} ===")
    for k in args.fields:
        if k == "dataset_reader":
            # make a fresh copy of the original data, so changes aren't reflected in the first dict
            tmp_data = copy.deepcopy(data)
            tmp_reader = {}
            # access the placeholder object
            placeholder_value = tmp_data["dataset_reader"]["readers"][TBID_PLACEHOLDER]
            # create tbid reader object and set attributes
            tmp_reader[tbid] = placeholder_value
            tmp_reader[tbid]["token_indexers"]["tokens"]["model_name"] = args.pretrained_model_name
            
            data["dataset_reader"]["readers"].update(tmp_reader)

            # delete placeholder
            if i == nb_tbids:
                del data["dataset_reader"]["readers"][TBID_PLACEHOLDER]

        elif k == "model":
            tmp_model = copy.deepcopy(data)

            # allowed arguments
            tmp_allowed_arguments = {}
            placeholder_value = tmp_data["model"]["allowed_arguments"][TBID_PLACEHOLDER]
            tmp_allowed_arguments[f"{tbid}_{args.head_type}"] = placeholder_value
            data["model"]["allowed_arguments"].update(tmp_allowed_arguments)

            # delete placeholder
            if i == nb_tbids:
                del data["model"]["allowed_arguments"][TBID_PLACEHOLDER]

            # backbone
            data["model"]["backbone"]["text_field_embedder"]["token_embedders"]["tokens"]["model_name"] = args.pretrained_model_name


            if args.model_type == "singleview":
                data["model"]["desired_order_of_heads"] = [f"{tbid}_{args.head_type}"]

        elif k == "heads":
            # make a fresh copy of the original data, so changes aren't reflected in the first dict
            tmp_data = copy.deepcopy(data)
            tmp_heads = {}
            # access the placeholder object
            placeholder_value = tmp_data["model"]["heads"][TBID_PLACEHOLDER]
            tmp_heads[f"{tbid}_{args.head_type}"] = placeholder_value
            data["model"]["heads"].update(tmp_heads)

            # delete placeholder
            if i == nb_tbids:
                del data["model"]["heads"][TBID_PLACEHOLDER]

            # if args.use_cross_stitch:
            #     # cross-stitch layer is in the meta head
            #     head_dict["multiview_meta_dependencies"]["use_cross_stitch"] = bool(f"{args.use_cross_stitch}")
            #     data["model"]["heads"].update(head_dict)

        elif k == "train_data_path":
            _file = f"{tbid}-ud-train.conllu"
            # we will just get the main paths once
            pathname = os.path.join(args.dataset_dir, "*", _file)
            train_path = glob.glob(pathname).pop()
            treebank_path = os.path.dirname(train_path)
            tmp_train = {tbid: f"{treebank_path}/{_file}"}
            data["train_data_path"].update(tmp_train)

            # delete placeholder
            if i == nb_tbids:
                del data["train_data_path"][TBID_PLACEHOLDER]
                del data["validation_data_path"][TBID_PLACEHOLDER]
                del data["test_data_path"][TBID_PLACEHOLDER]

        elif k == "validation_data_path":
            _file = f"{tbid}-ud-dev.conllu"
            tmp_dev = {tbid: f"{treebank_path}/{_file}"}
            data["validation_data_path"].update(tmp_dev)

        elif k == "test_data_path":
            _file = f"{tbid}-ud-test.conllu"
            tmp_test = {tbid: f"{treebank_path}/{_file}"}
            data["test_data_path"].update(tmp_test)

        elif k == "validation_metric":
            if args.model_type == "singleview":
                metric = f"+{tbid}_dependencies_LAS"
            elif args.model_type == "multiview":
                metric = f"+meta_dependencies_LAS"
            data["trainer"]["validation_metric"] = metric

        elif k == "random_seed":
            data["random_seed"] = args.random_seed
        elif k == "numpy_seed":
            data["numpy_seed"] = args.random_seed[:-1]
        elif k == "pytorch_seed":
            data["pytorch_seed"] = args.random_seed[:-2]

# write out custom config
with open(f"{dest_file}", "w") as fo:
    json.dump(data, fo, indent=2)
    print(json.dumps(data, indent = 2))


cmd = f"allennlp train -f {dest_file} -s {logdir} --include-package multiview_parser"
print("\nLaunching training script!")
rcmd = subprocess.call(cmd, shell=True)
