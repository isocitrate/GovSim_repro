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

# Read files and convert
band4 = rasterio.open("/project/macs30123/landsat8/LC08_B4.tif") #red
band5 = rasterio.open("/project/macs30123/landsat8/LC08_B5.tif") #nir
red = band4.read(1).astype("float64")
nir = band5.read(1).astype("float64")

# Save original dimensions
row, col = red.shape

# Convert to vec
red_flat = np.ascontiguousarray(red.flatten())
nir_flat = np.ascontiguousarray(nir.flatten())

# Set kernel logic
ndvi_kernel = ElementwiseKernel(
    "double *red, double *nir, double *ndvi",
    "ndvi[i] = (nir[i] - red[i]) / (nir[i] + red[i])",
    "ndvi_kernel",
    options=["-ccbin", "/usr/bin/gcc", "-allow-unsupported-compiler"]
)

# Time and run the GPU code
t0 = time.time()

# Throw to GPU
red_gpu = gpuarray.to_gpu(red_flat)
nir_gpu = gpuarray.to_gpu(nir_flat)
ndvi_gpu = gpuarray.empty_like(red_gpu)

# Run kernel
ndvi_kernel(red_gpu, nir_gpu, ndvi_gpu)

# Pull back to CPU
ndvi = ndvi_gpu.get()
t1 = time.time()
print(f"Time taken for GPU code: {t1 - t0:.2f} seconds")


# Time old CPU code
ndvi_cpu = (nir - red) / (nir + red)
t2 = time.time()
print(f"Time taken for CPU code: {t2 - t1:.2f} seconds")

# Save
ndvi = ndvi.reshape((row, col))
plt.imsave("q3a.png", ndvi, cmap="RdYlGn")