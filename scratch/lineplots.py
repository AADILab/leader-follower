import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sample data - List of paths, where each path is a list of (x, y) tuples
paths = [
    [(0, 0), (1, 1), (2, 0), (3, -1), (0,3)],
    [(0, 1), (1, 0), (2, 1), (3, 2), (0,3)],
    [(0, -1), (1, -2), (2, -1), (3, 0), (0,3)]
]

# Prepare data for Seaborn
data = []
for i, path in enumerate(paths):
    for x, y in path:
        data.append({'x': x, 'y': y, 'Path': f'Path {i+1}'})

df = pd.DataFrame(data)

# Set Seaborn style and color palette
sns.set(style="darkgrid")
palette = sns.color_palette('tab10', n_colors=len(paths))

# Create the plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='x', y='y', hue='Path', palette=palette)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Multiple Paths with Different Colors')
plt.legend()
plt.show()

