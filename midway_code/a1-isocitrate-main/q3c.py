# Import required libraries
import rasterio
import numpy as np
import time
import math
import matplotlib.pyplot as plt

import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel

# Hack to make compiler work
import pycuda.compiler
pycuda.compiler.DEFAULT_NVCC_FLAGS = ['-ccbin', '/usr/bin/gcc','-allow-unsupported-compiler']

# Load files and set up
sizes = [50, 100, 150]

ndvi_kernel = ElementwiseKernel(
    "double *red, double *nir, double *ndvi",
    "ndvi[i] = (nir[i] - red[i]) / (nir[i] + red[i])",
    "ndvi_kernel",
    options=["-ccbin", "/usr/bin/gcc", "-allow-unsupported-compiler"]
)

band4 = rasterio.open("/project/macs30123/landsat8/LC08_B4.tif") #red
band5 = rasterio.open("/project/macs30123/landsat8/LC08_B5.tif") #nir
red = band4.read(1).astype("float64")
nir = band5.read(1).astype("float64")


for size in sizes:
    red_plus = np.tile(red, size)
    nir_plus = np.tile(nir, size)

    row, col = red_plus.shape

    # Time the CPU
    t0 = time.time()
    ndvi_cpu = (nir_plus - red_plus) / (nir_plus + red_plus)
    t1 = time.time()
    print(f"Time taken for CPU code: {t1 - t0:.3f} seconds")

    # Time the GPU
    red_flat = np.ascontiguousarray(red_plus.flatten())
    nir_flat = np.ascontiguousarray(nir_plus.flatten())

    red_gpu = gpuarray.to_gpu(red_flat)
    nir_gpu = gpuarray.to_gpu(nir_flat)
    ndvi_gpu = gpuarray.empty_like(red_gpu)

    ndvi_kernel(red_gpu, nir_gpu, ndvi_gpu)

    ndvi = ndvi_gpu.get()
    t2 = time.time()
    print(f"Time taken for GPU code (size: {size}): {t2 - t1:.3f} seconds")





