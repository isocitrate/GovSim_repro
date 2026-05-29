import numpy as np
import scipy.stats as sts
from numba import jit
import time
from mpi4py import MPI

# Set params and load function from Q1a
rho = 0.5
mu = 3.0
sigma = 1.0
z_0 = mu
S = 1000
T = int(4160)

"""
@jit(nopython = True)
def simulateLife_jit(S, T, rho, mu, z_0, eps_mat, z_mat):
    for s_ind in range(S):
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t
"""
from test_module import simulateLife_aot

# Init. MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
np.random.seed(rank)

# Define the parallel structure in this core
nS = S // size
remainder = S % size
# Distribute the remainder in the first few ranks
nS = nS + 1 if rank < remainder else nS

eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, nS))
z_mat = np.zeros((T, nS))

# Wait til all cores ready to start
comm.Barrier() 
t0 = time.time()

# Do the simulation
simulateLife_aot(nS, T, rho, mu, z_0, eps_mat, z_mat)

# Wait til all cores done
comm.Barrier() 
t1 = time.time()


# Print results
if rank == 0:
    # Printing in a CSV-friendly format (cores, time) makes plotting easier later
    print(f"{size},{(t1 - t0):.3f}")