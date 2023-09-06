import sys; sys.path.append("/home/gonzaeve/boids/leader-follower")
from lib.stats_helpers import get1DSweepStatistics
import matplotlib.pyplot as plt
import numpy as np
import os

num_groups = np.array([1,5,10,15,20,25, 50, 100])

plot_min_max_range = False

sweep_stats_G, sweep_stats_D, sweep_stats_Df, sweep_stats_Z = get1DSweepStatistics(
    num_batches = len(num_groups),
    last_trial_num = 462, 
    num_stat_runs = 3, 
    computer_name = "silver-sabre",
    tested_Zero = False,
    tested_Dfollow = True, 
    tested_D = False,
    tested_G = True
)

legend = []
plt.plot(num_groups, sweep_stats_G["avg_team_fitness_arr"], color='tab:blue')
legend.append("$G$")

# plt.plot(num_groups, sweep_stats_D["avg_team_fitness_arr"], color='tab:orange')
# legend.append("$D$")

plt.plot(num_groups, sweep_stats_Df["avg_team_fitness_arr"], color='tab:green')
legend.append(r"$D_{follow}$")

# plt.plot(num_groups, sweep_stats_Z["avg_team_fitness_arr"], color='tab:pink')
# legend.append("$Zero$")

plt.fill_between(num_groups, sweep_stats_G["upper_err_team_fitness_arr"], sweep_stats_G["lower_err_team_fitness_arr"], alpha=0.2, color='tab:blue')
# plt.fill_between(num_groups, sweep_stats_D["upper_err_team_fitness_arr"], sweep_stats_D["lower_err_team_fitness_arr"], alpha=0.2, color='tab:orange')
plt.fill_between(num_groups, sweep_stats_Df["upper_err_team_fitness_arr"], sweep_stats_Df["lower_err_team_fitness_arr"], alpha=0.2, color='tab:green')
# plt.fill_between(num_groups, sweep_stats_Z["upper_err_team_fitness_arr"], sweep_stats_Z["lower_err_team_fitness_arr"], alpha=0.2, color='tab:pink')

if plot_min_max_range:
    plt.fill_between(num_groups, sweep_stats_G["upper_range"], sweep_stats_G["lower_range"], alpha=0.2, color='tab:blue')
    # plt.fill_between(num_groups, sweep_stats_D["upper_range"], sweep_stats_D["lower_range"], alpha=0.2, color='tab:orange')
    plt.fill_between(num_groups, sweep_stats_Df["upper_range"], sweep_stats_Df["lower_range"], alpha=0.2, color='tab:green')
    # plt.fill_between(num_groups, sweep_stats_Z["upper_range"], sweep_stats_Z["lower_range"], alpha=0.2, color='tab:pink')


plt.legend(legend)
plt.xlabel("Number of Leaders:Followers:POIs")
plt.ylabel("Performance")

plt.grid()
plt.ylim([0.,1.])
plt.xticks(num_groups)
figname = "1_1_1_sweep_trials_415_to_462"
if plot_min_max_range:
    figname += " | full range.png"
else:
    figname += " | std err.png"
plt.savefig(os.path.join("figures", figname))
plt.show()
