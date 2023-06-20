import sys; sys.path.append("/home/gonzaeve/boids/leader-follower")
from lib.stats_helpers import get1DSweepStatistics
import matplotlib.pyplot as plt
import numpy as np
import os

couplings = np.array([1,2,3])

plot_min_max = True

sweep_stats_G, sweep_stats_D, sweep_stats_Df, sweep_stats_Z = get1DSweepStatistics(
    num_batches = 3,
    last_trial_num = 414, 
    num_stat_runs = 10, 
    computer_name = "silver-sabre",
    tested_Zero = True,
    tested_Dfollow = True, 
    tested_D = True,
    tested_G = True
)

legend = []
plt.plot(couplings, sweep_stats_G["avg_team_fitness_arr"], color='tab:blue')
legend.append("$G$")

plt.plot(couplings, sweep_stats_D["avg_team_fitness_arr"], color='tab:orange')
legend.append("$D$")

plt.plot(couplings, sweep_stats_Df["avg_team_fitness_arr"], color='tab:green')
legend.append(r"$D_{follow}$")

plt.plot(couplings, sweep_stats_Z["avg_team_fitness_arr"], color='tab:pink')
legend.append("$Zero$")

plt.fill_between(couplings, sweep_stats_G["upper_err_team_fitness_arr"], sweep_stats_G["lower_err_team_fitness_arr"], alpha=0.2, color='tab:blue')
plt.fill_between(couplings, sweep_stats_D["upper_err_team_fitness_arr"], sweep_stats_D["lower_err_team_fitness_arr"], alpha=0.2, color='tab:orange')
plt.fill_between(couplings, sweep_stats_Df["upper_err_team_fitness_arr"], sweep_stats_Df["lower_err_team_fitness_arr"], alpha=0.2, color='tab:green')
plt.fill_between(couplings, sweep_stats_Z["upper_err_team_fitness_arr"], sweep_stats_Z["lower_err_team_fitness_arr"], alpha=0.2, color='tab:pink')

plt.legend(legend)
plt.xlabel("Coupling Requirement")
plt.ylabel("Performance")

plt.grid()
plt.ylim([0.,1.])
plt.xticks(couplings)
plt.savefig(os.path.join("figures/coupling_sweep_trials_295_to_414.png"))
plt.show()
