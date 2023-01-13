# todo
# from copy import deepcopy
# from leader_follower.file_helper import getLatestTrialName, loadConfig, loadTrial
# from leader_follower.watch_helpers import watchTeam
# from leader_follower.ccea_lib import CCEA
#
# # Loading in default config
# config = loadConfig()
#
# save_data = loadTrial(getLatestTrialName())
# team_data = save_data["best_team_data"]
#
# # watchTeam(config, team_data)
#
# # Create a CCEA for perturbing weights
# ccea = CCEA(**config["CCEA"])
#
# original_genome = deepcopy(team_data.team[0].genome)
#
# # Watch possible mutations of genome with loaded config
# num_mutations = 10
# for n in range(num_mutations):
#     # Create a mutation
#     team_data.team[0].genome = ccea.mutateGenome(original_genome)
#     # Watch the mutated genome
#     watchTeam(config, team_data)
