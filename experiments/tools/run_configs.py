# Specify directory
# Specify config number to start at
# Specify config number to end at

import argparse

from lib.file_helper import loadConfig, loadConfigData
from lib.learn_helpers import runExperiment

parser = argparse.ArgumentParser(description="Run specified configurations.")
parser.add_argument('experiment_name', type=str, help="Folder for configurations")
parser.add_argument('start_num', type=int, help="Which config to start with (inclusive)")
parser.add_argument('end_num',type=int, help="Which config to end at (inclusive)")
parser.add_argument('num_workers', type=int, help="Number of threads to run for ccea evaluations")
parser.add_argument('-l', '--legacy', action='store_true')

args = parser.parse_args()

# Generate list of trial numbers
trial_list = [args.start_num]
while trial_list[-1] < args.end_num:
    next_trial = trial_list[-1]+1
    trial_list.append(next_trial)

# Run all of those trials
for trial_num in trial_list:
    if args.legacy:
        config = loadConfig(computername=args.experiment_name, config_name="config_"+str(trial_num)+".yaml")
    else:
        config = loadConfigData(trialname="trial_"+str(trial_num), computername=args.experiment_name)
    config["CCEA"]["num_workers"] = args.num_workers
    runExperiment(config, computername=args.experiment_name, trial_num=str(trial_num), save_trial_only=False)
