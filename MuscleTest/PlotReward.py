import numpy as np
import matplotlib.pyplot as plt

reward = np.loadtxt("reacher.txt")
reward = reward.reshape((int(reward.shape[0]/2),2))
plt.plot(reward)
plt.show()

