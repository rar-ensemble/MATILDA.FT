import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('bonds')
plt.plot(data[:,0],data[:,1])
plt.xlabel("Time")
plt.ylabel("Bonded fraction (%)")
plt.show()