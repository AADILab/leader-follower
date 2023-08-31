from lib.ccea_lib import CCEA
from time import time
import sys
from typing import Dict, Optional
import numpy as np
from lib.file_helper import saveTrial, loadConfig, setupInitialPopulation, generateTrialPath, saveConfig, getLatestTrialNum

def runExperiment(config: Dict, computername: Optional[str] = None, trial_num: Optional[str] = None, save_trial_only: bool=False) -> None:
    # Start clock
    start = time()

    # Setup learner
    if trial_num is None:
        # The next trial is the latest trial plus 1
        trial_num = str(int(getLatestTrialNum(computername=computername))+1)

    learner = CCEA(**config["CCEA"], init_population=setupInitialPopulation(config), trial_path=generateTrialPath(computername, trial_num))
    try:
        learner.train(num_generations=config["num_generations"])
    except KeyboardInterrupt:
        print("Program interrupted by user keyboard interrupt. Exiting program and saving experiment data.")

    learner.stop_event.set()

    best_fitness_list, best_fitness_list_unfiltered, best_agent_fitness_lists_unfiltered,\
        average_fitness_list_unfiltered, average_agent_fitness_lists_unfiltered,\
        final_population, finished_iterations, best_team_data, \
        teams_in_evaluations, populations_through_generations, final_evaluation_teams = learner.getFinalMetrics()

    # Save data
    # save_data = {
    #     "scores_list": best_fitness_list,
    #     "unfiltered_scores_list": best_fitness_list_unfiltered,
    #     "unfiltered_agent_scores_list": best_agent_fitness_lists_unfiltered,
    #     "average_fitness_list_unfiltered": average_fitness_list_unfiltered,
    #     "average_agent_fitness_lists_unfiltered": average_agent_fitness_lists_unfiltered,
    #     "final_population": final_population,
    #     "finished_iterations": finished_iterations,
    #     "best_team_data": best_team_data,
    #     "teams_in_evaluations": teams_in_evaluations,
    #     "populations_through_generations": populations_through_generations,
    #     "final_evaluation_teams": final_evaluation_teams
    # }

    # saveTrial saves both the save data and the config
    # Let's set it up to just save the trial if specified
    # Also let's propogate computer name and the trial number down to here as well
    # Without breaking any existing software infrastructure

    saveConfig(config=config, computername=computername, trial_num=trial_num, folder_save=True)
    # saveTrial(save_data, config, computername=computername, trial_num=trial_num, save_trial_only=save_trial_only)

    print("Experiment time: ", time() - start, " seconds. Completed ", finished_iterations, " out of ", config["num_generations"], " generations.")

