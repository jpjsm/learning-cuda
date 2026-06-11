"""_summary_
## Gemini Overview
PyCUDA is a popular Python wrapper that allows developers to access NVIDIA's
CUDA parallel computation API directly from Python. It bridges the gap between
Python's productivity and the raw performance of GPU computing by letting you
write CUDA C/C++ kernels and execute them using Python scripts.

## Key Features

- **NumPy Integration**: Seamlessly transfers data between Python and the GPU
via NumPy arrays using the `pycuda.gpuarray.GPUArray class`.
- **Metaprogramming**: You can generate and compile CUDA code dynamically using
Python, which provides massive flexibility for building optimized runtime
configurations.
- **Automatic Memory Management**: Ties GPU object cleanup to Python object
lifecycles.
- **Error Handling**: Automatically converts CUDA errors into standard Python
exceptions.

## How to Get Started

You can easily install PyCUDA using pip.
Ensure that your NVIDIA driver and the CUDA Toolkit are already installed and
correctly configured on your system.
```bash
pip install pycuda
```

Use code with caution.

Once installed, a standard PyCUDA workflow involves:

- Allocating memory on the GPU.
- Compiling your CUDA C++ code in Python using SourceModule.
- Copying host (CPU) data to the device (GPU).
- Launching the kernel.
- Copying the result back to the host.

## Example Code

Below is a simple example showing how to add two arrays together on the GPU
using PyCUDA:
"""

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np

# 1. Generate data on the CPU
a = np.random.randn(400).astype(np.float32)
b = np.random.randn(400).astype(np.float32)

# 2. Allocate memory on the GPU and transfer data
a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
dest_gpu = gpuarray.empty_like(a_gpu)

# 3. Define the CUDA kernel in C++
mod = SourceModule("""
__global__ void add_vectors(float *dest, float *a, float *b) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    dest[i] = a[i] + b[i];
}
""")

# 4. Get the compiled function from the module and launch it
func = mod.get_function("add_vectors")
func(dest_gpu, a_gpu, b_gpu, block=(400, 1, 1), grid=(1, 1))

# 5. Fetch and verify the results
dest = dest_gpu.get()
assert np.allclose(dest, a + b)
print("Success! GPU and CPU results match.")
