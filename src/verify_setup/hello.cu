#include <stdio.h>

__global__ void hello()
{
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main()
{
    hello<<<1, 8>>>();
    cudaDeviceSynchronize();
    return 0;
}