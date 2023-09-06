import sys; sys.path.append("/home/gonzaeve/boids/leader-follower")
from plotting.plot_prelim import getStatistics
import matplotlib.pyplot as plt

num_leaders_list = [1,2]
num_pois_list = [1,2,3]
coupling_list = [1,2,3,4,5]
num_stat_runs = 20

end_trial = 4580


def getCombinationResults(num_leaders, num_pois, coupling):
    # Return the final
    # average, upper err, lower err for D, G, Df for that combination
    
    # First get the right trial number for the last trial of that combination
    # for the coupling

    # If the coupling is 5, then it's just the last one
    # if the coupling is 4, then it's the last one - (3*2*60)
    # if the coupling is 3, then it's the last one - 2*(3*2*60)
    coupling_trial_num = end_trial - (5-coupling)*(3*2*60)
    
    # Same idea for num_leaders
    # If the num_leaders is 2, then it's just the last one
    # If the num_leaders is 1, then it's the last one - (3*60)
    num_leader_trial_num = coupling_trial_num - (2-num_leaders)*(3*60)
    
    # Same idea for pois
    # If the num_pois is 3, then it's just the last one
    # If the num pois is 2, then its the last one - (60)
    # if the num pois is 1, then its the last one - 2*60
    num_pois_trial_num = num_leader_trial_num - (3-num_pois)*(60)

    trial_num = num_pois_trial_num

    # Get the statistics for Df
    trials_Dfollow = []
    for _ in range(num_stat_runs):
        trials_Dfollow.append("trial_"+str(trial_num))
        trial_num -= 1

    trials_D = []    
    for _ in range(num_stat_runs):
        trials_D.append("trial_"+str(trial_num))
        trial_num -= 1
    
    trials_G = []
    for _ in range(num_stat_runs):
        trials_G.append("trial_"+str(trial_num))
        trial_num -= 1        

    avg_G, std_err_G, upper_err_G, lower_err_G, upper_range_G, lower_range_G = getStatistics(trials_G)
    avg_D, std_err_D, upper_err_D, lower_err_D, upper_range_D, lower_range_D = getStatistics(trials_D)
    avg_Df, std_err_Df, upper_err_Df, lower_err_Df, upper_range_Df, lower_range_Df = getStatistics(trials_Dfollow)

    return avg_G[-1], upper_err_G[-1], lower_err_G[-1], avg_D[-1], upper_err_D[-1], lower_err_D[-1], avg_Df[-1], upper_err_Df[-1], lower_err_Df[-1]

def plotSweep(num_leaders, num_pois):
    # Specify number of leaders
    # Specify number of pois
    # Plot coupling on x axis
    # Plot average performance of different reward shaping methods on the y axis

    avg_G_list = []
    upper_G_list = []
    lower_G_list = []

    avg_D_list = []
    upper_D_list = []
    lower_D_list = []

    avg_Df_list = []
    upper_Df_list = []
    lower_Df_list = []

    for coupling_val in coupling_list:
        avg_G, upper_G, lower_G, avg_D, upper_D, lower_D, avg_Df, upper_Df, lower_Df = getCombinationResults(num_leaders=num_leaders, num_pois=num_pois, coupling=coupling_val)
        
        avg_G_list.append(avg_G)
        upper_G_list.append(upper_G)
        lower_G_list.append(lower_G)
    
        avg_D_list.append(avg_D)
        upper_D_list.append(upper_D)
        lower_D_list.append(lower_D)

        avg_Df_list.append(avg_Df)
        upper_Df_list.append(upper_Df)
        lower_Df_list.append(lower_Df)
    
    plt.figure()
    plt.ylim([0,1.0])
    legend = ["$G$", "$D$", r"$D_{follow}$"]

    plt.plot(coupling_list, avg_G_list, color='tab:blue')
    plt.plot(coupling_list, avg_D_list, color='tab:orange')
    plt.plot(coupling_list, avg_Df_list, color='tab:green')

    plt.fill_between(coupling_list, upper_G_list, lower_G_list, color='tab:blue', alpha=0.2)
    plt.fill_between(coupling_list, upper_D_list, lower_D_list, color='tab:orange', alpha=0.2)
    plt.fill_between(coupling_list, upper_Df_list, lower_Df_list, color='tab:green', alpha=0.2)

    plt.legend(legend)

    plt.xlabel("Coupling Requirement")
    plt.ylabel("Performance")

    plt.xticks(coupling_list)

    title = f"Num Leaders = {num_leaders} | Num POIs = {num_pois}"
    plt.title(title)

    plt.savefig("figures/"+title+".png")

# getCombinationResults(num_leaders=2,num_pois=2,coupling=1)
for num_leaders in num_leaders_list:
    for num_pois in num_pois_list:
        plotSweep(num_leaders, num_pois)
