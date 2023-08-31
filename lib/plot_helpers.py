import matplotlib.pyplot as plt
from lib.data_helpers import getEvalStatistics, getBestStatistics
import numpy as np
from typing import Optional
from enum import IntEnum

class PerformanceMetric(IntEnum):
    BestTrainingTeam = 0
    EvaluationTeam = 1

def plotStatisticsAvg(avg, color):
    num_generations_arr = np.arange(avg.shape[0])
    plt.plot(num_generations_arr, avg, color=color)

def plotStatisticsRange(upper_dev, lower_dev, upper_range, lower_range, color, plot_min_max_range=False):
    num_generations_arr = np.arange(upper_dev.shape[0])
    plt.fill_between(num_generations_arr, upper_dev.flatten(), lower_dev.flatten(), alpha=0.2, color=color)
    if plot_min_max_range:
        plt.fill_between(num_generations_arr, upper_range.flatten(), lower_range.flatten(), alpha=0.2, color=color)

def plotBatchPerformance(trial_datas_G: Optional[dict], trial_datas_D: Optional[dict], trial_datas_Dfollow: Optional[dict], \
                        plot_min_max_range: bool, start_trial_num: int, num_stat_runs: int, computername: str, performance_metric: PerformanceMetric):
    plt.figure(0)

    if performance_metric.value == PerformanceMetric.BestTrainingTeam.value:
        getStatistics = getBestStatistics
        title = "Best Training Team"
    elif performance_metric.value == PerformanceMetric.EvaluationTeam.value:
        getStatistics = getEvalStatistics
        title = "Evaluation Team"

    # Get statistics for different reward structures
    # We plot the baselines first so that D-Indirect is on the top layer
    legend = []
    if trial_datas_G is not None:
        avg_G, std_dev_G, upper_err_G, lower_err_G, upper_G, lower_G = getStatistics(trial_datas_G)
        plotStatisticsAvg(avg_G, color='tab:blue')
        legend.append("$G$")
        num_generations_arr = np.arange(avg_G.shape[0])

    if trial_datas_D is not None:
        avg_D, std_dev_D, upper_err_D, lower_err_D, upper_D, lower_D = getStatistics(trial_datas_D)
        plotStatisticsAvg(avg_D, color='tab:orange')
        legend.append("$D$")
        num_generations_arr = np.arange(avg_D.shape[0])

    if trial_datas_Dfollow is not None:
        avg_Df, std_dev_Df, upper_err_Df, lower_err_Df, upper_Df, lower_Df = getStatistics(trial_datas_Dfollow)
        plotStatisticsAvg(avg_Df, color='tab:green')
        legend.append(r'$D^I$')
        num_generations_arr = np.arange(avg_Df.shape[0])

    # Automatically figure out how many generations were in here
    plt.ylim([0,1.01])
    plt.xlim([0,len(num_generations_arr)-1])

    # Add the standard error or min max plotting
    if trial_datas_G is not None: 
        plotStatisticsRange(upper_err_G, lower_err_G, upper_G, lower_G, 'tab:blue', plot_min_max_range)
    if trial_datas_D is not None:
        plotStatisticsRange(upper_err_D, lower_err_D, upper_D, lower_D, 'tab:orange', plot_min_max_range)
    if trial_datas_Dfollow is not None:
        plotStatisticsRange(upper_err_Df, lower_err_Df, upper_Df, lower_Df, 'tab:green', plot_min_max_range)

    plt.legend(legend)

    # plt.legend(["$G$", "$D$", r'$D_{follow}$'])

    plt.xlabel("Number of Generations")
    plt.ylabel("Performance")
    plt.title(title)

    # plt.xlim([0,150])

    plot_save_name = "figures/trial_"+str(start_trial_num)+" | stat_runs "+str(num_stat_runs)+" | "+title+" |"
    if trial_datas_G:
        plot_save_name += " G"
    if trial_datas_D:
        plot_save_name += " D"
    if trial_datas_Dfollow:
        plot_save_name += " Df"
    if plot_min_max_range:
        plot_save_name += " | full range"
    else:
        plot_save_name += " | std err"
    plot_save_name += " | " + computername
    plot_save_name += ".png"

    print("Saving plot as ", plot_save_name)
    plt.savefig(plot_save_name)

    plt.show()
