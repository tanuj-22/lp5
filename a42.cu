#include<iostream>
#include<cuda.h>

__global__ void matrixMul(int *a, int *b, int *c, int n){

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < n && col < n){
        int sum = 0;
        // c[i][j] = a[i][0]*b[0][j] + a[i][1]*b[1][j] + ... + a[i][n-1]*b[n-1][j]
        for(int k=0;k<n;k++){
            sum += a[row*n + k] * b[k*n + col];
        }
        c[row*n + col] = sum;
    }
}


int main(){

    int size = 4;
    int a[size][size];
    int b[size][size];
    int c[size][size];

    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
            a[i][j] = i+j;
            b[i][j] = i-j;
        }
    }

    // declare
    int *a_gpu,*b_gpu,*c_gpu;
    cudaMalloc(&a_gpu,size*size*sizeof(int));
    cudaMalloc(&b_gpu,size*size*sizeof(int));
    cudaMalloc(&c_gpu,size*size*sizeof(int));


    // transfer
    cudaMemcpy(a_gpu,a,size*size*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu,b,size*size*sizeof(int),cudaMemcpyHostToDevice);

    int block_size = 16;
    dim3 dimBlock(block_size,block_size);
    dim3 dimGrid((size + dimBlock.x + 1)/ dimBlock.x,(size + dimBlock.y + 1)/ dimBlock.y);

    matrixMul<<<dimGrid,dimBlock>>>(a_gpu,b_gpu,c_gpu,size);

    cudaMemcpy(c,c_gpu,size*size*sizeof(int),cudaMemcpyDeviceToHost);

    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
            std::cout<<c[i][j]<<" ";
        }
        std::cout<<std::endl;
    }

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);

    return 0;

}






























// This program demonstrates array addition using CUDA. The arrayAddition kernel function is responsible for performing the addition of corresponding elements from arrays a and b, and storing the result in array c. Each thread is assigned an element to compute.

// In the main function, the host arrays a, b, and c are initialized. Memory is allocated on the device using cudaMalloc, and the input arrays are copied from the host to the device using cudaMemcpy.

// The grid and block dimensions are defined to configure how the CUDA threads are organized. In this example, each block contains threadsPerBlock number of threads, and the number of blocks is calculated based on the array size.

// The arrayAddition kernel is launched using <<<blocksPerGrid, threadsPerBlock>>>, passing the device arrays and size as arguments.

// After the kernel execution, the result array c is copied back from the device to the host using cudaMemcpy, and the final result is printed.

// Finally, memory allocated on the device is freed using cudaFree.

// Note: Make sure to compile this program with the appropriate CUDA compiler and link against the CUDA libraries.


// CUDA is a parallel computing platform and application programming interface that allows software to use certain types of graphics processing units for general purpose processing, an approach called general-purpose computing on GPUs.
// o(n^2.373)
