import cupy as cp

x = cp.arange(10)
y = x * 2

print("CuPy works:", y)

# Get number of CUDA devices
num_devices = cp.cuda.runtime.getDeviceCount()
# Display device IDs and names
for i in range(num_devices):
    with cp.cuda.Device(i):
        name = cp.cuda.runtime.getDeviceProperties(i)["name"].decode("utf-8")
        print(f"Device {i}: {name}")
