# Configs 247 - 291. Sweep of different numbers of learners to followers

import sys; sys.path.append("/home/egonzalez/leaders")
import numpy as np
import matplotlib.pyplot as plt
from lib.file_helper import loadConfig, loadTrial

start_trial = 247
num_var = 15

configs = ["config_"+str(i+start_trial)+".yaml" for i in range(num_var*3)]
trials = ["trial_"+str(i+start_trial) for i in range(num_var*3)]

# Get the average best performance of each one
save_datas = [loadTrial(trial) for trial in trials]
best_fitnesses = [save_data["best_team_data"].fitness for save_data in save_datas]
averages = [np.average(best_fitnesses[(3*i):(3*i)+3]) for i in range(num_var)]
uppers = [averages[i]+np.std(best_fitnesses[(3*i):(3*i)+3]) for i in range(num_var)]
lowers = [averages[i]-np.std(best_fitnesses[(3*i):(3*i)+3]) for i in range(num_var)]

# Get the recorded ratio of learners to followers
config_dicts = [loadConfig(config) for config in configs]
num_followers = [config_dict["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"]  for config_dict in config_dicts][::3]
# ratios = [(15.-num_follow)/num_follow for num_follow in num_followers]

plt.figure(0)
plt.ylim([0.0, 1.0])

plt.fill_between(x=num_followers,y1=lowers, y2=uppers, alpha=0.2)
plt.plot(num_followers, averages)
plt.xticks(num_followers)
plt.ylabel("System Performance")
plt.xlabel("Number of Followers")
plt.title("Varying Number of Followers out of 15 agents.")
plt.show()
