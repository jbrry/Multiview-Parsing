"""
Script to launch the multiview parser.
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
parser.add_argument(
    "--config",
    default="configs/multiview/dependency_parser_multiview.jsonnet",
    type=str,
    help="Configuration file",
)
parser.add_argument(
    "--dataset-dir",
    default="data/ud-treebanks-v2.8-sampled",
    type=str,
    help="The path containing all UD treebanks",
)
parser.add_argument(
    "--dataset-dir-concatenated",
    default="data/ud-treebanks-v2.8-concatenated",
    type=str,
    help="The path containing concatenated UD treebanks",
)
parser.add_argument(
    "--results-dir",
    default="results",
    type=str,
    help="Where to write results",
)
parser.add_argument(
    "--tbids", default=[], type=str, nargs="+", help="Treebank ids to include"
)
parser.add_argument(
    "--model-type",
    choices=["singleview", "singleview-concat", "multiview"],
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
    "--recover",
    default=False,
    action="store_true",
    help="Flag to recover training"
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
    "--desired-batch-size",
    type=int,
    default=32,
    help="Batch size for experiment, will be used to calculate steps for grad accum.",
)
parser.add_argument(
    "--gpu-batch-size",
    type=int,
    default=8,
    help="Batch size on a physical device, hardware-bound.",
)
parser.add_argument(
    "--num-gpus",
    type=int,
    default=1,
    help="Number of GPUs.",
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
        "data_loader",
        "model",
        "heads",
        "train_data_path",
        "validation_data_path",
        "test_data_path",
        "validation_metric",
        "random_seed",
        "trainer",
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

if args.use_cross_stitch:
    cross_stitch_string = "-cross-stitch"
else:
    cross_stitch_string = ""

run_name = original_json_file[0] + f"-{model_identifier}-" + "+".join(args.tbids) + metadata_string + cross_stitch_string + "-" + args.random_seed
config_name = run_name + original_json_file[1]

logdir = f"logs/dependency_parsing_multiview/{run_name}"

dest.append(config_name)
dest_file = "/".join(dest)

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

nb_tbids = len(args.tbids)


class ExperimentBuilder:
    def __init__(self, args):
        self.args = args
        self.tbids = args.tbids
        self.model_type = args.model_type
        self.head_type = args.head_type
        self.fields = args.fields
        self.recover = args.recover
        self.use_cross_stitch = args.use_cross_stitch
        self.pretrained_model_name = args.pretrained_model_name
        self.dataset_dir = args.dataset_dir
        self.dataset_dir_concatenated = args.dataset_dir_concatenated
        self.random_seed = args.random_seed
        self.results_dir = args.results_dir
        self.desired_batch_size =  args.desired_batch_size
        self.gpu_batch_size = args.gpu_batch_size
        self.num_gpus = args.num_gpus

        self.GDRIVE_DEST="gdrive:Parsing-PhD-Folder/UD-Data/cross-lingual-parsing/Multilingual_Parsing_Meta_Structure/experiment_logs/"

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        
    def prepare_and_run_experiment(self):
        print(f"creating config {run_name}")

        for i, tbid in enumerate(self.tbids, start=1):
            print(f"=== {tbid} ===")
            for k in self.fields:
                if k == "dataset_reader":
                    # make a fresh copy of the original data, so changes aren't reflected in the first dict
                    tmp_data = copy.deepcopy(data)
                    tmp_reader = {}
                    # access the placeholder object
                    placeholder_value = tmp_data["dataset_reader"]["readers"][TBID_PLACEHOLDER]
                    # create tbid reader object and set attributes
                    tmp_reader[tbid] = placeholder_value
                    tmp_reader[tbid]["token_indexers"]["tokens"]["model_name"] = self.pretrained_model_name
                    
                    data["dataset_reader"]["readers"].update(tmp_reader)

                    # delete placeholder
                    if i == nb_tbids:
                        del data["dataset_reader"]["readers"][TBID_PLACEHOLDER]

                elif k == "model":
                    tmp_model = copy.deepcopy(data)

                    # allowed arguments
                    tmp_allowed_arguments = {}
                    placeholder_value = tmp_data["model"]["allowed_arguments"][TBID_PLACEHOLDER]
                    tmp_allowed_arguments[f"{tbid}_{self.head_type}"] = placeholder_value
                    data["model"]["allowed_arguments"].update(tmp_allowed_arguments)

                    # delete placeholder
                    if i == nb_tbids:
                        del data["model"]["allowed_arguments"][TBID_PLACEHOLDER]

                    # backbone
                    data["model"]["backbone"]["text_field_embedder"]["token_embedders"]["tokens"]["model_name"] = self.pretrained_model_name

                    if self.model_type == "singleview":
                        data["model"]["desired_order_of_heads"].append(f"{tbid}_{args.head_type}")
                    elif self.model_type == "multiview":
                        data["model"]["desired_order_of_heads"].append(f"{tbid}_{args.head_type}")
                    
                        if i == nb_tbids:
                            data["model"]["desired_order_of_heads"].append(f"multi_dependencies")
                            data["model"]["desired_order_of_heads"].append(f"meta_dependencies")
                        
                elif k == "heads":
                    # make a fresh copy of the original data, so changes aren't reflected in the first dict
                    tmp_data = copy.deepcopy(data)
                    tmp_heads = {}
                    # access the placeholder object
                    placeholder_value = tmp_data["model"]["heads"][TBID_PLACEHOLDER]
                    tmp_heads[f"{tbid}_{args.head_type}"] = placeholder_value
                    data["model"]["heads"].update(tmp_heads)

                    # delete placeholder and check meta head
                    if i == nb_tbids:
                        del data["model"]["heads"][TBID_PLACEHOLDER]
                        if self.model_type == "multiview" and self.use_cross_stitch:
                            data["model"]["heads"]["meta_dependencies"]["use_cross_stitch"] = bool(True)
                           
                elif k == "train_data_path":
                    _file = f"{tbid}-ud-train.conllu"
                    # we will just get the main paths once
                    if self.model_type == "singleview-concat":
                        pathname = os.path.join(self.dataset_dir_concatenated, "*", _file)
                    else:
                        pathname = os.path.join(self.dataset_dir, "*", _file)

                    train_path = glob.glob(pathname).pop()
                    treebank_path = os.path.dirname(train_path)
                    
                    if os.path.isfile(f"{treebank_path}/{_file}"):
                        tmp_train = {tbid: f"{treebank_path}/{_file}"}
                        data["train_data_path"].update(tmp_train)

                    # delete placeholder
                    if i == nb_tbids:
                        del data["train_data_path"][TBID_PLACEHOLDER]
                        if not data["train_data_path"]:
                            raise ValueError("No training data")

                elif k == "validation_data_path":
                    _file = f"{tbid}-ud-dev.conllu"
                    if os.path.isfile(f"{treebank_path}/{_file}"):
                        print(data["validation_data_path"])
                        tmp_dev = {tbid: f"{treebank_path}/{_file}"}
                        data["validation_data_path"].update(tmp_dev)
                
                    if i == nb_tbids:
                        del data["validation_data_path"][TBID_PLACEHOLDER]
                        if not data["validation_data_path"]:
                            del data["validation_data_path"]

                elif k == "test_data_path":
                    _file = f"{tbid}-ud-test.conllu"
                    if os.path.isfile(f"{treebank_path}/{_file}"):
                        tmp_test = {tbid: f"{treebank_path}/{_file}"}
                        data["test_data_path"].update(tmp_test)

                    if i == nb_tbids:
                        del data["test_data_path"][TBID_PLACEHOLDER]
                        if not data["test_data_path"]:
                            del data["test_data_path"]
                
                elif k == "data_loader":
                    if i == nb_tbids:
                        data["data_loader"]["scheduler"]["batch_size"] = self.gpu_batch_size

                elif k == "trainer":
                    if i == nb_tbids:
                        effective_batch_size = self.gpu_batch_size * self.num_gpus
                        num_gradient_accumulation_steps = self.desired_batch_size / effective_batch_size
                        data["trainer"]["num_gradient_accumulation_steps"] = int(num_gradient_accumulation_steps)

                elif k == "validation_metric":
                    if self.model_type == "singleview": # or "singleview-concat":
                        self.metric = f"+{tbid}_dependencies_LAS"
                    elif self.model_type == "singleview-concat":
                        self.metric = f"+{tbid}_dependencies_LAS"
                    elif self.model_type == "multiview":
                        self.metric = f"+meta_dependencies_LAS"
                    
                    if i == nb_tbids:
                        data["trainer"]["validation_metric"] = self.metric

                elif k == "random_seed":
                    data["random_seed"] = self.random_seed
                elif k == "numpy_seed":
                    data["numpy_seed"] = self.random_seed[:-1]
                elif k == "pytorch_seed":
                    data["pytorch_seed"] = self.random_seed[:-2]

        # write out custom config
        with open(f"{dest_file}", "w") as fo:
            json.dump(data, fo, indent=2)
            print(json.dumps(data, indent=2))

        # if no tar file and no logdir, then try download it.
        if not os.path.exists(f"{logdir}.tar.gz") and not os.path.exists(f"{logdir}"):
            cmd = f"rclone copy {self.GDRIVE_DEST}{run_name}.tar.gz {os.path.dirname(logdir)}"
            rcmd = subprocess.call(cmd, shell=True)

        # if we downloaded a file, extract it
        if os.path.exists(f"{logdir}.tar.gz"):
            cmd = f"tar -xvzf {logdir}.tar.gz"
            rcmd = subprocess.call(cmd, shell=True)
        
        # check if the tar contains a model
        if os.path.isfile(f"{logdir}/model.tar.gz"):
            print(f"skipping training, file exists.")
        else:
            if not self.recover:
                # remove tarred file which only contains stdout etc.
                print("removing tarred file with no model")
                cmd = f"rm {logdir}.tar.gz"
                rcmd = subprocess.call(cmd, shell=True)
                
                print("training from scratch")
                cmd = f"allennlp train -f {dest_file} -s {logdir} --include-package multiview_parser"
            else:
                # if we're recovering, we need to make sure there is a valid model state/training state to recover from.
                # the next bit of code searches for model states and checks if there is a corresponding training state.
                # if not, the model state will be deleted, falling back to an earlier state we can train from.
                model_files = []
                for f in os.listdir(f"{logdir}"):
                    # if we have a model state, check if we also have a training state. If not delete it
                    if "model_state" in f:
                        model_files.append(f)

                for mf in sorted(model_files): # sort on epoch number
                    epoch_item = mf.split("_")[2] # ['model', 'state', 'e46', 'b0.th']
                    training_state = f"training_state_{epoch_item}_b0.th"
                    # if there's a training state, do nothing
                    if os.path.exists(f"{logdir}/{training_state}"):
                        continue
                    else: # else, we need to delete the model state
                        print(f"removing model file with no training state: {logdir}/{mf}")
                        os.remove(f"{logdir}/{mf}")
                
                print("recovering training")
                cmd = f"allennlp train -s {logdir} -r --include-package multiview_parser {dest_file}"

            print("\nLaunching training script!")
            rcmd = subprocess.call(cmd, shell=True)

        # predict
        if self.model_type  == "singleview":
            self.predict_singleview()
        elif self.model_type == "singleview-concat":
            self.predict_singleview_concat()
        elif self.model_type == "multiview":
            self.predict_multiview()


        # tar the file (in the case we've just trained a new model) or if we recovered and need to create a new one.
        if not os.path.exists(f"{logdir}.tar.gz") or self.recover:
            cmd = f"tar -cvzf {logdir}.tar.gz {logdir}/"
            print(f"tarring {tbid}")
            rcmd = subprocess.call(cmd, shell=True)
            # store it on Google Drive
            cmd = f"rclone copy {logdir}.tar.gz {self.GDRIVE_DEST}"
            print(f"uploading {tbid}")
            rcmd = subprocess.call(cmd, shell=True)

        # clean-up directory
        cmd = f"rm -r {logdir} {logdir}.tar.gz"
        print(f"cleaning up")
        rcmd = subprocess.call(cmd, shell=True)


    def predict_singleview(self):
        """Returns the command to run the predictor with the appropriate inputs/outputs."""
        print("predicting singleview")
        for tbid in args.tbids:
            predictor_args = '{"head_name": ""}'
            predictor_json = json.loads(predictor_args)
            predictor_json["head_name"] = f"{tbid}_dependencies"
            predictor_json_string = json.dumps(predictor_json)
            
            target_files = []

            validation_data = data.get("validation_data_path", None)
            if not validation_data:
                print("no dev data found")
            else:
                target_file = data["validation_data_path"][tbid]
                target_files.append(target_file)

            test_data = data.get("test_data_path", None)
            if not test_data:
                print("no test data found")
            else:
                target_file = data["test_data_path"][tbid]
                target_files.append(target_file)

            for target_file in target_files:
                filename_short = os.path.splitext(os.path.basename(target_file))[0]
                if "dev" in filename_short:
                    mode = "dev"
                elif "test" in filename_short:
                    mode = "test"

                outfile = f"{self.results_dir}/{run_name}-ud-{mode}.conllu"
                result_evalfile = f"{self.results_dir}/{run_name}-ud-{mode}-eval.txt"
                
                cmd = f"allennlp predict {logdir}/model.tar.gz {target_file} \
                    --output-file {outfile} \
                    --predictor conllu-multitask-predictor \
                    --include-package multiview_parser \
                    --use-dataset-reader \
                    --predictor-args '{predictor_json_string}' \
                    --multitask-head {tbid} \
                    --batch-size 32 \
                    --cuda-device 0 \
                    --silent"

                print(f"predicting {tbid} in mode {mode}")
                rcmd = subprocess.call(cmd, shell=True)
                cmd = f"python scripts/conll18_ud_eval.py -v {target_file} {outfile} > {result_evalfile}"
                print(f"evaluating {tbid}")
                rcmd = subprocess.call(cmd, shell=True)


    def predict_singleview_concat(self):
        """
        Returns the command to run the predictor with the appropriate inputs/outputs.
        Has some extra logic to loop over concatenated tbids to predict on the individual tbids.
        """
        print("predicting singleview-concat")
        for tbid in args.tbids:
            predictor_args = '{"head_name": ""}'
            predictor_json = json.loads(predictor_args)
            predictor_json["head_name"] = f"{tbid}_dependencies"
            predictor_json_string = json.dumps(predictor_json)

            for sub_tbid in tbid.split("+"):
                print(f"predicting {sub_tbid}")
                # check if the files exist
                train_file = f"{sub_tbid}-ud-train.conllu" # may not always have dev
                pathname = os.path.join(args.dataset_dir, "*", train_file) # NOTE: using non-concat dir
                train_path = glob.glob(pathname).pop()
                treebank_path = os.path.dirname(train_path)
                
                for mode in ["dev", "test"]:
                    print(f"predicting {mode}")
                    _file = f"{sub_tbid}-ud-{mode}.conllu"
                    if os.path.isfile(f"{treebank_path}/{_file}"):
                        outfile = f"{self.results_dir}/{run_name}-{sub_tbid}-ud-{mode}.conllu"
                        result_evalfile = f"{self.results_dir}/{run_name}-{sub_tbid}-ud-{mode}-eval.txt"
                        target_file =  f"{treebank_path}/{sub_tbid}-ud-{mode}.conllu"

                        cmd = f"allennlp predict {logdir}/model.tar.gz {target_file} \
                            --output-file {outfile} \
                            --predictor conllu-multitask-predictor \
                            --include-package multiview_parser \
                            --use-dataset-reader \
                            --predictor-args '{predictor_json_string}' \
                            --multitask-head {tbid} \
                            --batch-size 32 \
                            --cuda-device 0 \
                            --silent"

                        print(f"predicting {tbid}")
                        rcmd = subprocess.call(cmd, shell=True)
                        cmd = f"python scripts/conll18_ud_eval.py -v {target_file} {outfile} > {result_evalfile}"
                        print(f"evaluating {tbid}")
                        rcmd = subprocess.call(cmd, shell=True)

    def predict_multiview(self):
        print("predicting multiview")
        # predicter args stay outside the loop as we are always using the meta head.
        predictor_args = '{"head_name": ""}'
        predictor_json = json.loads(predictor_args)
        predictor_json["head_name"] = "meta_dependencies"
        predictor_json_string = json.dumps(predictor_json)
        
        for tbid in args.tbids:
            target_files = []
            # Need to try here as the value will be present from obtaining the data from the previous tbid.
            # In the singleview case it should work fine there because there is just one iteration.
            try:
                target_file = data["validation_data_path"][tbid]
                target_files.append(target_file)
            except KeyError:
                print("no dev data found")

            try:
                target_file = data["test_data_path"][tbid]
                target_files.append(target_file)
            except KeyError:
                print("no test data found")

            for target_file in target_files:
                print(target_file)
                filename_short = os.path.splitext(os.path.basename(target_file))[0]
                if "dev" in filename_short:
                    mode = "dev"
                elif "test" in filename_short:
                    mode = "test"

                outfile = f"{self.results_dir}/{run_name}-{tbid}-ud-{mode}.conllu"
                result_evalfile = f"{self.results_dir}/{run_name}-{tbid}-ud-{mode}-eval.txt"
                
                cmd = f"allennlp predict {logdir}/model.tar.gz {target_file} \
                    --output-file {outfile} \
                    --predictor conllu-multitask-predictor \
                    --include-package multiview_parser \
                    --use-dataset-reader \
                    --predictor-args '{predictor_json_string}' \
                    --multitask-head {tbid} \
                    --batch-size 32 \
                    --cuda-device 0 \
                    --silent"

                print(f"predicting {tbid} in mode {mode}")
                rcmd = subprocess.call(cmd, shell=True)
                cmd = f"python scripts/conll18_ud_eval.py -v {target_file} {outfile} > {result_evalfile}"
                print(f"evaluating {tbid}")
                rcmd = subprocess.call(cmd, shell=True)


experiment_builder = ExperimentBuilder(args)
experiment = experiment_builder.prepare_and_run_experiment()
