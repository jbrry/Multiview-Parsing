import json
import logging
import collections
import math
import csv
import os
import sys


def main():

    # main log dir / results directory
    logdir = sys.argv[1]
    outdir = sys.argv[2]
    metric = sys.argv[3]

    KEYS = [f"validation_{metric}", f"validation_character_{metric}", f"validation_meta_{metric}", f"validation_word_{metric}"]
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    csv_file = open(f"{outdir}/experiment_results.csv", "w")
    csv_writer = csv.writer(csv_file)
    # we will write out the run name, the epoch, the metric we are tracking and its value
    out_data = ["run", "epoch", "metric", "value"]
    header = out_data
    csv_writer.writerow(header)

    d_completed = {}

    for run in os.listdir(logdir):
        print(run)
        completed = False

        path = os.path.join(logdir, run)
        dbucket = {}
        for _file in os.listdir(path):
            #print(_file)
            if _file.startswith("metrics_epoch"):
                string, end = _file.split(".")
                epoch = string.split("_")[-1]
                # to permit lexographic sorting
                if len(epoch) == 1:
                    epoch = str(0) + epoch
                
                dbucket[epoch] = _file
                        
            # Check for complete run
            elif _file == "metrics.json":
                completed = True
            
        d_completed[run] = completed

        # sort bucket on epoch number
        sbucket = collections.OrderedDict(sorted(dbucket.items()))
        #print(sbucket)
        # collect data from the sorted metrics files
        for epoch_number, metrics_file in sbucket.items():
            metrics_path = os.path.join(logdir, run, metrics_file)
            with open(metrics_path) as json_file:
                data = json.load(json_file)
                for key in data:
                    if key in KEYS:
                        # important row: store run, epoch, metric and value
                        row = []
                        row.append(run)
                        row.append(epoch_number)
                        row.append(key)
                        row.append(data[key])
                        print(row)

                        csv_writer.writerow(row)
    
    print(d_completed)

if __name__ == '__main__':
    main()

