import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("q2a_out.csv", delimiter = ",")

plt.scatter(data[:, 0], data[:, 1])
plt.xlabel("Rho")
plt.ylabel("Average period to negative health")
plt.savefig("q2b_plot.png", dpi=300)