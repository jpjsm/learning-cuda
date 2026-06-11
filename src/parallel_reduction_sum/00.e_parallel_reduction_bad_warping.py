import sys
from typing import List

import cupy as cp

kernel_code = r"""
// ------------------------------
// Integer type trait replacement
// ------------------------------
template<typename T> struct is_int { static const bool value = false; };

template<> struct is_int<int> { static const bool value = true; };
template<> struct is_int<unsigned int> { static const bool value = true; };
template<> struct is_int<long long> { static const bool value = true; };
template<> struct is_int<unsigned long long> { static const bool value = true; };

// ------------------------------
// Integer power (binary exponentiation)
// ------------------------------
template <typename Tbase, typename Texp>
__device__ long long intPow(Tbase base, Texp exp) {
    // Compile-time type checking without <type_traits>
    static_assert(is_int<Tbase>::value, "Base must be an integer type");
    static_assert(is_int<Texp>::value, "Exponent must be an integer type");

    // No exceptions allowed in NVRTC handle manually
    if (exp < 0) {
        return 0;  // or define your own behavior
    }

    long long result = 1;
    long long b = (long long)base;
    unsigned long long e = (unsigned long long)exp;

    while (e > 0) {
        if (e & 1ULL) {
            result *= b;
        }
        b *= b;
        e >>= 1;
    }
    return result;
}


extern "C" __global__ 
void reduce_single_kernel_warp(
    long long* numbers, 
    long long count)
{
    unsigned long long tid = (unsigned long long) threadIdx.x;
    unsigned long long block = (unsigned long long) blockIdx.x;
    unsigned long long block_size = (unsigned long long) blockDim.x;
    
    unsigned int lane = (unsigned int)(threadIdx.x % 32); // lane index within warp
    
    long long sum = 0;

    int round = 0;
    long long stride = intPow(32, round);
    long long index = (block * block_size + tid) * stride;
    
    
    while (stride < count)
    {        
        // Warp-level reduction using shuffle down
        sum = numbers[index];
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (lane == 0)
        {
            numbers[index] = sum;            
        }
                
        round++;
        stride = intPow(32, round);
        index = (block * block_size + tid) * stride;
    }

}
"""

my_kernel = cp.RawModule(code=kernel_code)
p_reduce_single_kernel_warp = my_kernel.get_function("reduce_single_kernel_warp")

DEBUG = False


def parallel_sum(numbers: List[int]) -> int:
    if not isinstance(numbers, (list, tuple)):
        raise ValueError("'numbers' must be a list")

    if len(numbers) == 0:
        raise ValueError("'numbers' must be a list with one or more elements")

    if not all(isinstance(x, int) and not isinstance(x, bool) for x in numbers):
        raise ValueError("'numbers' must be a list with integer type elements")

    N = len(numbers)
    threads = 1024
    blocks = (N + threads - 1) // threads
    gpu_numbers = cp.array(numbers, dtype=cp.int64)
    print(f"[INFO]«parallel_sum()» {N=:14,} | {threads=:6,} | {blocks=:26,}")
    p_reduce_single_kernel_warp((blocks,), (threads,), (gpu_numbers, N))

    return gpu_numbers[0].item()


if __name__ == "__main__":
    list_size_powers = list(range(6, 31, 4))
    for power in list_size_powers:
        size = 2**power
        expected = size * (size - 1) // 2
        numbers = list(range(size))
        actual = parallel_sum(numbers)
        print(
            f"[INFO]«main()          » {power=:3,} | {size=:14,} | {expected=:26,} | {actual=:26,} | Are Equal? {actual==expected}\n\n"
        )
