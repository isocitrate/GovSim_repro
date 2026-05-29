import numpy as np
import scipy.stats as sts
from numba import jit
from numba.pycc import CC
import time

# Set model parameters
rho = 0.5
mu = 3.0
sigma = 1.0
z_0 = mu

# Set simulation parameters, draw all idiosyncratic random shocks,
# and create empty containers

S = 1000 # Set the number of lives to simulate
T = int(4160) # Set the number of periods for each simulation

np.random.seed(25)
eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))
z_mat = np.zeros((T, S))

def simulateLife(S, T, rho, mu, z_0, eps_mat, z_mat):
    for s_ind in range(S):
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t

@jit(nopython = True)
def simulateLife_jit(S, T, rho, mu, z_0, eps_mat, z_mat):
    for s_ind in range(S):
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t

cc = CC("test_module")
@cc.export('simulateLife_aot', '(i4, i4, f4, f4, f4, f8[:,:], f8[:,:])')
def simulateLife_aot(S, T, rho, mu, z_0, eps_mat, z_mat):
    for s_ind in range(S):
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t

cc.compile()


from test_module import simulateLife_aot

t0 = time.time()
simulateLife(S, T, rho, mu, z_0, eps_mat, z_mat)
t1 = time.time() 
simulateLife_jit(S, T, rho, mu, z_0, eps_mat, z_mat)
t2= time.time() 
simulateLife_aot(S, T, rho, mu, z_0, eps_mat, z_mat)
t3 = time.time() 

print(f"Time taken for raw Python, JIT, and AOT are, respectively: {(t1-t0):.3f}, {(t2-t1):.3f}, {(t3-t2):.3f}")