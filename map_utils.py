from turtle import pos
import numpy as np

class Map():
    def __init__(self, map_size, observation_radius, positions=[]):
        self.map_size = map_size
        self.observation_radius = observation_radius
        self.num_bins = self.calculate_num_bins()
        self.bins = self.populate_bins(positions)

    def reset(self, positions=[]):
        self.bins = self.populate_bins(positions)

    def calculate_num_bins(self):
        return np.ceil(self.map_size/self.observation_radius).astype(int)

    def create_bins(self):
        bins = np.frompyfunc(list, 0, 1)(np.empty(self.num_bins, dtype=object))
        return bins

    def calculate_bin_location(self, position):
        # Calculate bin location for this agent with no bound (nb)
        bin_location_nb = (position/self.observation_radius).astype(int)-1
        # Bound bin location for edge case where agent is on top and/or right edge of map
        bin_location = np.minimum(bin_location_nb, self.num_bins-1)
        # Bound bin location for edge case where agent is on bottom and/or left edge of map
        bin_location = np.maximum(bin_location_nb, [0,0])
        return bin_location

    def populate_bins(self, positions):
        bins = self.create_bins()
        print("bins:\n", bins)
        for ind, position in enumerate(positions):
            print("position: ", position)
            # Calculate the bin location of this position
            bin_location = self.calculate_bin_location(position)
            print("bin_location: ", bin_location)
            # print(position)
            # print(bins.shape)
            # print(bin_location)
            # print(bins[bin_location[0], bin_location[1]])
            # Put the index in the bin
            bins[bin_location[0], bin_location[1]].append(ind)
            print("bins:\n", bins)
        return bins

    def get_adj_bins(self, bin_location):
        adj_bins = bin_location + np.array([
            [0,0],
            [0,1],
            [1,1],
            [1,0],
            [1,-1],
            [0,-1],
            [-1,-1],
            [-1,0],
            [-1,1]
        ])
        # Make sure returned bins are valid through logic operations
        above_origin = np.greater(adj_bins, 0)
        below_bounds = np.less(adj_bins, self.bins.shape)
        valid_ind = np.logical_and(above_origin, below_bounds)
        valid_adj_bins = adj_bins[valid_ind]
        return valid_adj_bins

    def get_adj_agent_inds(self, position):
        bin_location = self.calculate_bin_location(position)
        adj_bins = self.get_adj_bins(bin_location)
        agent_inds = []
        for bin in adj_bins:
            agent_inds += self.bins[bin[0], bin[1]]
        return agent_inds

    def get_observable_agent_inds(self, position, positions):
        # Get indicies of all agents in adjacent bins
        adj_agent_inds = self.get_adj_agent_inds(position)
        # Calculate how far away all of those agents are from specified position
        delta_pos = positions - position # Vectors from position to positions
        r = np.sqrt(delta_pos[:,0]**2 + delta_pos[:,1]**2)
        # Keep only the ones that are within the observation radius
        observable_ind_mask = np.less(r, positions[adj_agent_inds])
        observable_agent_inds = adj_agent_inds[observable_ind_mask]
        return observable_agent_inds
