import numpy as np


stations_pos = np.array([[0, 0], [5, 4], [4, 7], [4, 0]])
# Sort stations_pos based on the first column, then the second column
sorted_indices = np.lexsort((stations_pos[:, 1], stations_pos[:, 0]))
stations_pos = stations_pos[sorted_indices]
print(stations_pos)