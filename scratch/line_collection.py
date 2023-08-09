import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

# Sample data
num_paths = 5
num_points = 100
print(np.linspace(0,10, int(num_points/2)))
x = np.hstack((np.linspace(0, 5, int(num_points/2)), np.linspace(2,0, int(num_points/2))))

paths_data = []

for i in range(num_paths):
    y = np.sin(x) + i
    paths_data.append(np.column_stack((x, y)))

# Create a list of line segments
lines = [path_data for path_data in paths_data]

# Create a LineCollection from the line segments
lc = LineCollection(lines, cmap=plt.get_cmap('tab10'), label=['Path {}'.format(i+1) for i in range(num_paths)])

# Set up the plot
plt.figure(figsize=(10, 6))
plt.gca().add_collection(lc)
plt.xlim(x.min(), x.max())
plt.ylim(np.min([path[:, 1].min() for path in paths_data]), np.max([path[:, 1].max() for path in paths_data]))
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Multiple Paths with Different Colors')
plt.legend()

# Show the plot
plt.show()
