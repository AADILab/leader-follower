import matplotlib.pyplot as plt
import numpy as np

# Sample data - List of paths, where each path is a list of (x, y) tuples
paths = [
    [(0, 0), (1, 1), (2, 0), (1, -1)],
    [(0, 1), (1, 0), (2, 1), (3, 2)],
    [(0, -1), (1, -2), (2, -1), (3, 0)]
]

# Set up the plot
plt.figure(figsize=(10, 6))
for path in paths:
    x, y = zip(*path)
    plt.plot(x, y)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Multiple Paths with Matplotlib')
plt.legend(['Path 1', 'Path 2', 'Path 3'])
plt.show()

