#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 32 * 1024 *1024
#define BLOCK_SIZE 256

__global__ void reduce_v0(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; // 计算当前线程需要处理的全局内存中数据的位置，其中blockIdx.x
                                                            // 是当前线程块在所有线程块中的索引，blockDim.x是线程块的宽度
    sdata[tid] = g_idata[i]; // 将全局内存中的数据加载到共享内存中
    __syncthreads(); // 确保所有线程都完成了上一步的加载数据操作

    // do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) { // 检查当前线程是否是每组2s个线程的第一个，如果是，则执行规约操作
            sdata[tid] += sdata[tid + s]; // 将当前线程的共享内存的值与它右侧第s个位置的值相加，实现规约
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

int main() {
    float *input_host = (float*)malloc(N * sizeof(float));
    float *input_device;
    cudaMalloc((void**)&input_device, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        input_host[i] = 2.0;
    }
    cudaMemcpy(input_device, input_host, N * sizeof(float), cudaMemcpyHostToDevice);

    int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float *output_host = (float*)malloc((N / BLOCK_SIZE) * sizeof(float));
    float *output_device;
    cudaMalloc((void **)&output_device, (N / BLOCK_SIZE) * sizeof(float));

    dim3 grid(N / BLOCK_SIZE, 1);
    dim3 block(BLOCK_SIZE, 1);
    reduce_v0<<<grid, block>>>(input_device, output_device);
    cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    return 0;
}