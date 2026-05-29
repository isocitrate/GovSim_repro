from mpi4py import MPI
import numpy as np
import scipy.stats as sts
import time
from numba import jit

# Init. MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Init. params and function
S = 1000
T = int(4160)
mu = 3.0
sigma = 1.0
z_0 = mu - 3 * sigma # New starting point for sick people

# Init. the containers
eps_mat = eps_mat = np.empty((T, S), dtype=np.float64)
rhos = np.empty(20, dtype=np.float64)
aves = np.empty(20, dtype=np.float64)

rhos_all = None
aves_all = None

# Init. the shock (once only)
if rank == 0:
    np.random.seed(rank)
    eps_mat[:, :] = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))

    # Also init. the big containers across all runs
    aves_all = np.empty(200, dtype = np.float64)
    rhos_all = np.linspace(-0.95, 0.95, 200)

# Cast shock to all
comm.Bcast(eps_mat, root = 0)
comm.Scatter(rhos_all, rhos, root = 0)

# Def the new function for each rho (breaks sim when sick)
@jit(nopython = True)
def simulateRho_sick(S, T, rho, mu, z_0, eps_mat):
    total = 0
    for s in range(S):
        z_tm1 = z_0
        life = T

        for t in range(T):
            e_t = eps_mat[t, s]
            z_t = rho * z_tm1 + (1-rho) * mu + e_t

            if z_t <= 0:
                life = t + 1
                break

            z_tm1 = z_t       
        total += life
    return total / S

# Compile first and set timer
_ = simulateRho_sick(1,1,0.5,mu, z_0, eps_mat)
t0 = time.time()

# Simulate for each rho using pre-compiled funciton
for i in range(20):
    aves[i] = simulateRho_sick(S, T, rhos[i], mu, z_0, eps_mat)

# Gather results back
comm.Gather(aves, aves_all, root = 0)

# Get optimal rho
if rank == 0:
    t1 = time.time()
    max_i = np.argmax(aves_all)
    max_ave = aves_all[max_i]
    rho_star = rhos_all[max_i]

    print(f"Optimal rho: {rho_star:.2f} with average life of {max_ave:.2f} periods, computed over {t1 - t0:.2f} seconds.")

    out = np.column_stack((rhos_all, aves_all))
    np.savetxt("q2a_out.csv", out, delimiter = ",")