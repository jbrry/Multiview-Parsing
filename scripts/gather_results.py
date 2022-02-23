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

# GROUP_TO_TBID_RESULTS = {
#     "af-de-nl": None,
#     "e-sla": None,
#     "en": None,
#     "es-ca": None,
#     "finno": None,
#     "fr": None,
#     "indic": None,
#     "iranian": None,
#     "it": None,
#     "ko": None,
#     "n-ger": None,
#     "old": None,
#     "pt-gl": None,
#     "sw-sla": None,
#     "turkic": None,
#     "w-sla": None,
# }


MODEL_TYPES = ["singleview", "singleview-concat", "multiview"]

GROUP_TO_TBID_RESULTS = {}
for group, tbids in GROUP_TO_TBID_MAPPINGS.items():
    GROUP_TO_TBID_RESULTS[group] = {}
    for model_type in MODEL_TYPES:
        GROUP_TO_TBID_RESULTS[group][model_type] = {}


for filename in os.listdir(args.result_dir):
    if filename.endswith("-dev-eval.txt"):
        # parse filename into fields:
        if "singleview" in filename:
            if "+" in filename:
                continue # skip concat for now
            else:
                model_type = "singleview"
                filename_short = os.path.splitext(filename)[0]
                filepath = os.path.join(args.result_dir, filename)
                x = filename_short.split("-")
                tbid = x[4]
                for group, tbids in GROUP_TO_TBID_MAPPINGS.items():
                    if tbid in tbids:
                        
                        if os.stat(filepath).st_size == 0:
                            print('File is empty')
                            GROUP_TO_TBID_RESULTS[group][model_type][tbid] = 0. 
                        else:
                            print('File is not empty')

                            with open(filepath, 'r') as fi:
                                for line in fi:
                                    items = line.split("|")
                                    if len(items) == 5:
                                        metric = items[METRIC].strip()
                                        if metric == "LAS":
                                            score = items[F1SCORE].strip()
                                            GROUP_TO_TBID_RESULTS[group][model_type][tbid] = score     
                                        


print(GROUP_TO_TBID_RESULTS)


import csv

with open('insights.csv', 'w', newline='') as csvfile:
    fieldnames = ['group', 'tbid'] + MODEL_TYPES
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for group, model_type in GROUP_TO_TBID_RESULTS.items():
        for mt in MODEL_TYPES:
            for tbid, score in model_type[mt].items():
                writer.writerow({'group': group, 'tbid': tbid, mt: score})

