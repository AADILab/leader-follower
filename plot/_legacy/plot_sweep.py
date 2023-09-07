import sys; sys.path.append("/home/gonzaeve/boids/leader-follower")
import numpy as np
import matplotlib.pyplot as plt
from lib.file_helper import loadConfig, loadTrial

start_trial = 179
num_var = 6

configs = ["config_"+str(i+start_trial)+".yaml" for i in range(num_var*3)]
trials = ["trial_"+str(i+start_trial) for i in range(num_var*3)]

# Get the average best performance of each one
save_datas = [loadTrial(trial) for trial in trials]
best_fitnesses = [save_data["best_team_data"].fitness for save_data in save_datas]
averages = [np.average(best_fitnesses[(3*i):(3*i)+3]) for i in range(num_var)]
uppers = [averages[i]+np.std(best_fitnesses[(3*i):(3*i)+3]) for i in range(num_var)]
lowers = [averages[i]-np.std(best_fitnesses[(3*i):(3*i)+3]) for i in range(num_var)]

# Get the recorded number of hidden units
config_dicts = [loadConfig(config) for config in configs]
hiddens = [config_dict["CCEA"]["nn_hidden"] for config_dict in config_dicts][::3]

plt.figure(0)
plt.ylim([0.0,1.0])
plt.fill_between(x=hiddens,y1=lowers, y2=uppers, alpha=0.2)
plt.plot(hiddens, averages)
plt.xticks(hiddens)
plt.ylabel("System Performance")
plt.xlabel("Number of Hidden Units")
plt.title("Varying Hidden Layer Size")
plt.show()
