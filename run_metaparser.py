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

from meta_parser import utils

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="", type=str, help="Log dir name")
parser.add_argument(
    "--config",
    default="configs/meta_parser/dependency_parser_source_transformer-target_word.jsonnet",
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
    choices=["source", "meta"],
    type=str,
    help="Model type: source-only is a pretrained transformer,"
    " meta is the meta model on top of the source and target models.",
)
parser.add_argument(
    "--do-lower-case",
    default=False,
    action="store_true",
    help="Whether to do lower case or not"
)
parser.add_argument(
    "--mono-transformer",
    default=False,
    action="store_true",
    help="Whether to use a mono transformer."
)
parser.add_argument(
    "--multi-transformer",
    default=False,
    action="store_true",
    help="Whether to use a multi transformer."
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

args = parser.parse_args()

jsonnet_file = args.config
data = json.loads(_jsonnet.evaluate_file(jsonnet_file))

basename = os.path.basename(args.config)
dirname = os.path.dirname(args.config)

dest = dirname.split("/")
dest.append("tmp")
dest_dir = "/".join(dest)

orig = os.path.splitext(basename)

# need to disambiguate between source-only runs with different PT LMs
model_identifier = args.pretrained_model_name.split("/")[-1]
if args.metadata:
    metadata_string = f"-{args.metadata}"
else:
    metadata_string = ""

run_name = orig[0] + f"-{model_identifier}-" + "-".join(args.tbids) + metadata_string
logdir = f"logs/meta_parser/{run_name}"
run_name = run_name + orig[1]

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
            placeholder = tmp_data["dataset_reader"]["readers"]["TBID_PLACEHOLDER"]
            tmp_reader = {tbid: placeholder}

            # multi
            try:
                multi_indexer = tmp_reader[tbid]["multi_token_indexers"]
       
                if args.multi_transformer:
                    # set model name
                    tmp_reader[tbid]["multi_token_indexers"]["tokens"][
                        "model_name"
                    ] = f"{args.pretrained_model_name}"

                    # set casing
                    tmp_reader[tbid]["multi_token_indexers"]["tokens"][
                        "tokenizer_kwargs"
                        ]["do_lower_case"] = bool(f"{args.do_lower_case}")
                else:
                    tmp_reader[tbid]["multi_token_indexers"]["tokens"][
                        "namespace"
                    ] = f"multi_tokens"   

            except KeyError:
                pass

            # token
            try:
                if args.mono_transformer:
                    tmp_reader[tbid]["mono_token_indexers"]["tokens"][
                        "model_name"
                    ] = args.pretrained_model_name
                else:
                    tmp_reader[tbid]["mono_token_indexers"]["tokens"][
                        "namespace"
                    ] = f"{tbid}_tokens"

            except KeyError:
                pass

            # token-char
            try:
                tmp_reader[tbid]["mono_token_character_indexers"]["token_characters"][
                    "namespace"
                ] = f"{tbid}_token_characters"
            except KeyError:
                pass

            # sentence-char
            try:
                tmp_reader[tbid]["mono_sentence_character_indexers"][
                    "token_characters"
                ]["namespace"] = f"{tbid}_sentence_characters"
            except KeyError:
                pass

            data["dataset_reader"]["readers"].update(tmp_reader)

        elif k == "model":
            tmp_data = copy.deepcopy(data)
            model_dict = tmp_data["model"]["backbone"]
            for k in model_dict.keys():
                # TODO: technically we could make the 'multi' view just a pretrained LM view,
                # make it wrap a monolingual LM for each langage,
                # and have the second view as multilingual chars/words or a dual monolingual view.
                # and the later layers share params
                # multi
                if k == "multi_embedder":
                    #tmp_embedder = {}
                    # get placeholder embedder
                    tmp_embedder = model_dict["multi_embedder"]

                    if args.multi_transformer:
                        tmp_embedder["token_embedders"]["tokens"][
                            "model_name"
                        ] = f"{args.pretrained_model_name}"
                    else:
                        tmp_embedder["token_embedders"]["tokens"][
                            "vocab_namespace"
                        ] = f"multi_tokens"

                    data["model"]["backbone"]["multi_embedder"] = tmp_embedder

                # token
                elif k == "mono_embedders":
                    tmp_embedder = {tbid: {}}
                    # get placeholder embedder
                    tmp_embedder[tbid] = model_dict["mono_embedders"][
                        "TBID_PLACEHOLDER"
                    ]
                    if args.mono_transformer:
                        # pretrained transformer
                        tmp_embedder[tbid]["token_embedders"]["tokens"][
                            "model_name"
                        ] = args.pretrained_model_name
                    else:
                        # regular embedder
                        tmp_embedder[tbid]["token_embedders"]["tokens"][
                            "vocab_namespace"
                        ] = f"{tbid}_tokens"

                    data["model"]["backbone"]["mono_embedders"].update(tmp_embedder)

                # token-char
                elif k == "mono_token_character_embedders":
                    tmp_embedder = {tbid: {}}

                    tmp_embedder[tbid] = model_dict["mono_token_character_embedders"][
                        "TBID_PLACEHOLDER"
                    ]
                    tmp_embedder[tbid]["token_embedders"]["token_characters"][
                        "vocab_namespace"
                    ] = f"{tbid}_token_characters"
                    data["model"]["backbone"]["mono_token_character_embedders"].update(
                        tmp_embedder
                    )

                # sentence-char
                elif k == "mono_sentence_character_embedders":
                    tmp_embedder = {tbid: {}}

                    tmp_embedder[tbid] = model_dict[
                        "mono_sentence_character_embedders"
                    ]["TBID_PLACEHOLDER"]
                    tmp_embedder[tbid]["token_embedders"]["token_characters"][
                        "vocab_namespace"
                    ] = f"{tbid}_sentence_characters"
                    data["model"]["backbone"][
                        "mono_sentence_character_embedders"
                    ].update(tmp_embedder)

                # monolingual encoders (do not have to be feature-specific)
                elif k == "mono_encoders":
                    tmp_encoder = {tbid: {}}
                    tmp_encoder[tbid] = model_dict["mono_encoders"]["TBID_PLACEHOLDER"]
                    data["model"]["backbone"]["mono_encoders"].update(tmp_encoder)

                # token-char
                elif k == "mono_token_character_encoders":
                    tmp_encoder = {tbid: {}}
                    tmp_encoder[tbid] = model_dict["mono_token_character_encoders"][
                        "TBID_PLACEHOLDER"
                    ]
                    data["model"]["backbone"]["mono_token_character_encoders"].update(
                        tmp_encoder
                    )

                # sentence-char
                elif k == "mono_sentence_character_encoders":
                    tmp_encoder = {tbid: {}}
                    tmp_encoder[tbid] = model_dict["mono_sentence_character_encoders"][
                        "TBID_PLACEHOLDER"
                    ]
                    data["model"]["backbone"][
                        "mono_sentence_character_encoders"
                    ].update(tmp_encoder)

        elif k == "heads":
            tmp_heads = copy.deepcopy(data)
            head_dict = tmp_heads["model"]["heads"]
            if args.use_cross_stitch:
                # cross-stitch layer is in the meta head
                head_dict["meta_dependencies"]["use_cross_stitch"] = bool(f"{args.use_cross_stitch}")
                data["model"]["heads"].update(head_dict)

        elif k == "train_data_path":
            _file = f"{tbid}-ud-train.conllu"
            # we will just get the main paths once
            pathname = os.path.join(args.dataset_dir, "*", _file)
            train_path = glob.glob(pathname).pop()
            treebank_path = os.path.dirname(train_path)
            tmp_train = {tbid: f"{treebank_path}/{_file}"}
            data["train_data_path"].update(tmp_train)

        elif k == "validation_data_path":
            _file = f"{tbid}-ud-dev.conllu"
            tmp_dev = {tbid: f"{treebank_path}/{_file}"}
            data["validation_data_path"].update(tmp_dev)

        elif k == "test_data_path":
            _file = f"{tbid}-ud-test.conllu"
            tmp_test = {tbid: f"{treebank_path}/{_file}"}
            data["test_data_path"].update(tmp_test)

        elif k == "validation_metric":

            if args.model_type == "source":
                metric = f"+multi_dependencies_LAS_{tbid}"
            elif args.model_type == "meta":
                metric = f"+meta_dependencies_LAS_AVG"

            data["trainer"]["validation_metric"] = metric

        elif k == "random_seed":
            data["random_seed"] = args.random_seed
        elif k == "numpy_seed":
            data["numpy_seed"] = args.random_seed[:-1]
        elif k == "pytorch_seed":
            data["pytorch_seed"] = args.random_seed[:-2]

    # delete placeholder entries at last tbid
    if i == nb_tbids:
        # dataset reader should encompass indexers
        del data["dataset_reader"]["readers"]["TBID_PLACEHOLDER"]
        try:
            tmp_dict = data["model"]["backbone"]["mono_embedders"]
            del data["model"]["backbone"]["mono_embedders"]["TBID_PLACEHOLDER"]
        except KeyError:
            pass

        # mono encoders
        try:
            tmp_dict = data["model"]["backbone"]["mono_encoders"]
            del data["model"]["backbone"]["mono_encoders"]["TBID_PLACEHOLDER"]
        except KeyError:
            pass

        # token-char
        try:
            tmp_dict = data["model"]["backbone"]["mono_token_character_embedders"]
            del data["model"]["backbone"]["mono_token_character_embedders"][
                "TBID_PLACEHOLDER"
            ]
        except KeyError:
            pass

        try:
            tmp_dict = data["model"]["backbone"]["mono_token_character_encoders"]
            del data["model"]["backbone"]["mono_token_character_encoders"][
                "TBID_PLACEHOLDER"
            ]
        except KeyError:
            pass

        # sentence-char
        try:
            tmp_dict = data["model"]["backbone"]["mono_sentence_character_embedders"]
            del data["model"]["backbone"]["mono_sentence_character_embedders"][
                "TBID_PLACEHOLDER"
            ]
        except KeyError:
            pass

        try:
            tmp_dict = data["model"]["backbone"]["mono_sentence_character_encoders"]
            del data["model"]["backbone"]["mono_sentence_character_encoders"][
                "TBID_PLACEHOLDER"
            ]
        except KeyError:
            pass

        # filepaths
        del data["train_data_path"]["TBID_PLACEHOLDER"]
        del data["validation_data_path"]["TBID_PLACEHOLDER"]
        del data["test_data_path"]["TBID_PLACEHOLDER"]

# write out custom config
with open(f"{dest_file}", "w") as fo:
    json.dump(data, fo, indent=2)

cmd = f"allennlp train -f {dest_file} -s {logdir} --include-package meta_parser"
print("Launching training script!")
rcmd = subprocess.call(cmd, shell=True)
