import json
import logging
import collections
import math
import csv
import os
import re
import sys
import glob
import shutil
import argparse
import subprocess

from collections import defaultdict


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("--output-dir", type=str, help="outputs")
parser.add_argument("--result-dir", default="results", type=str,
                    help="The path containing the results")
parser.add_argument("--treebank-ids", default=[], type=str, nargs="+",
                    help="Specify a list of treebank IDs to use")

args = parser.parse_args()

METRIC, PRECISION, RECALL, F1SCORE, ALIGNEDACC = range(5)

GROUP_TO_TBID_MAPPINGS = {
    "af-de-nl": ["af_afribooms", "nl_alpino", "nl_lassysmall", "de_gsd"],
    "e-sla": ["ru_syntagrus", "ru_taiga", "uk_iu"],
    "en": ["en_ewt", "en_gum", "en_lines"],
    "es-ca": ["ca_ancora", "es_ancora"],
    "finno": ["et_edt", "fi_ftb", "fi_tdt", "sme_giella"],
    "fr": ["fr_gsd", "fr_sequoia", "fr_spoken"],
    "indic": ["hi_hdtb", "ur_udtb"],
    "iranian": ["kmr_mg", "fa_seraji"],
    "it": ["it_isdt", "it_postwita"],
    "ko": ["ko_gsd", "ko_kaist"],
    "n-ger": ["da_ddt", "no_bokmaal", "no_nynorsk", "no_nynorsklia", "sv_lines", "sv_talbanken"],
    "old": ["grc_proiel", "grc_perseus", "got_proiel", "la_ittb", "la_proiel", "la_perseus", "cu_proiel"],
    "pt-gl": ["gl_ctg", "gl_treegal", "pt_bosque"],
    "sw-sla": ["hr_set", "sr_set", "sl_ssj", "sl_sst"],
    "turkic": ["bxr_bdt", "kk_ktb", "tr_imst", "ug_udt"],
    "w-sla": ["cs_cac", "cs_fictree", "cs_pdt", "pl_lfg", "pl_pdb", "sk_snk", "hsb_ufal"],
}


MODEL_TYPES = ["singleview", "singleview-concat", "multiview"]

NO_DEV = ["gl_treegal", "la_perseus", "hsb_ufal", "sme_giella", "kmr_mg", "kk_ktb", "bxr_bdt", "sl_sst"]

# populate results based on groups/model types
GROUP_TO_TBID_RESULTS = {}
for group, tbids in GROUP_TO_TBID_MAPPINGS.items():
    GROUP_TO_TBID_RESULTS[group] = {}
    for model_type in MODEL_TYPES:
        GROUP_TO_TBID_RESULTS[group][model_type] = {}

for filename in os.listdir(args.result_dir):
    if filename.endswith("-dev-eval.txt"):
        # parse filename into fields:
       
        if "singleview" in filename:
            # single-view concat
            if "+" in filename:
                model_type = "singleview-concat"
                filename_short = os.path.splitext(filename)[0]
                filepath = os.path.join(args.result_dir, filename)

                x = filename_short.split("-")
                tbid = x[-4]
                for group, tbids in GROUP_TO_TBID_MAPPINGS.items():
                    if tbid in tbids:
                        
                        if os.stat(filepath).st_size == 0:
                            print(f"File {filename_short} is empty")
                            if tbid in NO_DEV:
                                GROUP_TO_TBID_RESULTS[group][model_type][tbid] = "no-dev" 
                            else:
                                GROUP_TO_TBID_RESULTS[group][model_type][tbid] = 0. 
                        else:
                            with open(filepath, 'r') as fi:
                                for line in fi:
                                    items = line.split("|")
                                    if len(items) == 5:
                                        metric = items[METRIC].strip()
                                        if metric == "LAS":
                                            score = items[F1SCORE].strip()
                                            GROUP_TO_TBID_RESULTS[group][model_type][tbid] = score
                        
            else:
                model_type = "singleview"
                filename_short = os.path.splitext(filename)[0]
                filepath = os.path.join(args.result_dir, filename)
                x = filename_short.split("-")
                tbid = x[4]
                for group, tbids in GROUP_TO_TBID_MAPPINGS.items():
                    if tbid in tbids:
                        if os.stat(filepath).st_size == 0:
                            print(f"File {filename_short} is empty")
                            if tbid in NO_DEV:
                                GROUP_TO_TBID_RESULTS[group][model_type][tbid] = "no-dev" 
                            else:
                                GROUP_TO_TBID_RESULTS[group][model_type][tbid] = 0. 
                        else:

                            with open(filepath, 'r') as fi:
                                for line in fi:
                                    items = line.split("|")
                                    if len(items) == 5:
                                        metric = items[METRIC].strip()
                                        if metric == "LAS":
                                            score = items[F1SCORE].strip()
                                            GROUP_TO_TBID_RESULTS[group][model_type][tbid] = score

        elif "multiview" in filename:
            model_type = "multiview"
            filename_short = os.path.splitext(filename)[0]
            filepath = os.path.join(args.result_dir, filename)
            x = filename_short.split("-")
            tbid = x[-4]

            for group, tbids in GROUP_TO_TBID_MAPPINGS.items():
                if tbid in tbids:
                    
                    if os.stat(filepath).st_size == 0:
                        print(f"File {filename_short} is empty")
                        if tbid in NO_DEV:
                            GROUP_TO_TBID_RESULTS[group][model_type][tbid] = "no-dev" 
                        else:
                            GROUP_TO_TBID_RESULTS[group][model_type][tbid] = 0. 
                    else:
                        with open(filepath, 'r') as fi:
                            for line in fi:
                                items = line.split("|")
                                if len(items) == 5:
                                    metric = items[METRIC].strip()
                                    if metric == "LAS":
                                        score = items[F1SCORE].strip()
                                        GROUP_TO_TBID_RESULTS[group][model_type][tbid] = score
                                        

# print()
# print(GROUP_TO_TBID_RESULTS)
# print()
# x=sorted(GROUP_TO_TBID_RESULTS.items())
# print(x)

# raise ValueError()

with open('results/insights.csv', 'w', newline='') as csvfile:
    fieldnames = ['group', 'tbid', 'singleview', 'singleview-concat', 'multiview'] #+ MODEL_TYPES
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for group, model_type in GROUP_TO_TBID_RESULTS.items():
        tbids = model_type["singleview"].keys()
        # write based on alphabetical order
        for tbid in sorted(tbids):

            try:
                singleview_score = model_type["singleview"][tbid]
            except KeyError:
                singleview_score = 0.

            try:
                singleview_concat_score = model_type["singleview-concat"][tbid]
            except KeyError:
                singleview_concat_score = 0.

            try:
                multiview_score = model_type["multiview"][tbid]
            except KeyError:
                multiview_score = 0.

            writer.writerow({'group': group,
                            'tbid': tbid,
                            'singleview': singleview_score,
                            'singleview-concat': singleview_concat_score,
                            'multiview': multiview_score,
                            })
 

# 4) store it on Google Drive
GDRIVE_DEST="gdrive:Parsing-PhD-Folder/UD-Data/cross-lingual-parsing/Multilingual_Parsing_Meta_Structure/experiment_results/Insights"
cmd = f"rclone copy results/insights.csv {GDRIVE_DEST}"
print(f"uploading csv")
rcmd = subprocess.call(cmd, shell=True)
