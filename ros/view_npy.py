from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
import numpy as np

data = np.load('./pointcloud/temp.npy')
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]


fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z)
plt.show()
