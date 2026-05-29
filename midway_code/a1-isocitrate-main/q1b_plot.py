import matplotlib.pyplot as plt
import numpy as np

results = np.loadtxt("q1b.out", delimiter = ",")

plt.bar(results[:, 0], results[:, 1])
plt.xlabel("# cores")
plt.ylabel("Time, s")
plt.savefig("q1b_plot.png", dpi=300)

