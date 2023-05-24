#include<iostream>  
#include<cuda.h>

__global__ void add(int *a, int *b, int *c,int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main(){

    int n = 10;
    int *a_cpu = new int[n];
    int *b_cpu = new int[n];
    int *c_cpu = new int[n];

    // fill arrays with random numbers
    for (int i = 0; i < n; i++) {
        a_cpu[i] = rand() % 100;
        b_cpu[i] = rand() % 100;
    }

    // declare GPU memory pointers
    int *a_gpu, *b_gpu, *c_gpu;

    // allocate GPU memory
    cudaMalloc(&a_gpu, n * sizeof(int));
    cudaMalloc(&b_gpu, n * sizeof(int));
    cudaMalloc(&c_gpu, n * sizeof(int));

    // transfer the array to the GPU
    cudaMemcpy(a_gpu, a_cpu, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b_cpu, n * sizeof(int), cudaMemcpyHostToDevice);


    // define grid and block size
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // launch the kernel
    add<<<gridSize, blockSize>>>(a_gpu, b_gpu, c_gpu, n);

    cudaMemcpy(c_cpu, c_gpu, n * sizeof(int), cudaMemcpyDeviceToHost);

    // print out the results
    for (int i = 0; i < n; i++) {
        std::cout << a_cpu[i] << " + " << b_cpu[i] << " = " << c_cpu[i] << std::endl;
    }

    // free GPU memory allocation
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);

    // free CPU memory allocation
    delete[] a_cpu;
    delete[] b_cpu;
    delete[] c_cpu;

    return 0;

}


































// The provided code is a CUDA program that performs element-wise addition of two arrays `a` and `b` on the GPU using parallel execution.

// Here's an explanation of the code:

// 1. The code includes necessary header files: `iostream` and `cuda.h`.

// 2. The `add` kernel function is defined. It takes three integer array pointers `a`, `b`, and `c`, along with an integer `n` representing the array size. The kernel is executed by each thread, where the `index` is calculated based on the thread and block indices. The if condition ensures that threads outside the array bounds do not perform any computation. Each thread adds the corresponding elements from `a` and `b` and stores the result in `c`.

// 3. In the `main` function:
//    - The array size `n` is set to 10.
//    - Three integer arrays `a_cpu`, `b_cpu`, and `c_cpu` are created on the host (CPU) to store the input arrays and the result.
//    - The arrays `a_cpu` and `b_cpu` are filled with random numbers.
//    - Pointers for GPU memory allocation, `a_gpu`, `b_gpu`, and `c_gpu`, are declared.
//    - Memory is allocated on the GPU using `cudaMalloc` for `a_gpu`, `b_gpu`, and `c_gpu`.
//    - The input arrays `a_cpu` and `b_cpu` are transferred from the CPU to the GPU using `cudaMemcpy`.
//    - The block size and grid size are set to 256 and `(n + blockSize - 1) / blockSize` respectively, to configure the number of threads.
//    - The `add` kernel is launched using `<<<gridSize, blockSize>>>`, passing the GPU array pointers and the size `n`.
//    - The result array `c_gpu` is copied back from the GPU to the CPU using `cudaMemcpy`.
//    - Finally, the result is printed by iterating over the arrays `a_cpu`, `b_cpu`, and `c_cpu`.

// 4. Memory allocated on the GPU is freed using `cudaFree`.

// 5. Memory allocated on the CPU is released using `delete[]`.

// The code demonstrates how to perform array addition in parallel using CUDA. It utilizes GPU parallelism to perform the addition operation on multiple elements simultaneously, potentially achieving better performance compared to a sequential CPU implementation.




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
