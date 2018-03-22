nsparse: Fast Sparse Matrix Library for GPU
======

## Versions
1.5 (Nov, 2017)  


## Introduction
Sparse matrix computation is a key kernel of many applications. This library provides first sparse matrix computation kernels including SpMV and SpGEMM. A new sparse matrix format called the Adaptive Multi-level Blocking (AMB) format improves the performance of sparse matrix vector multiplication (SpMV). More information regarding AMB format can be found in (1). A new algorithm for Sparse General Matrix-Matrix Multiplication (SpGEMM) improves the FLOPS performance with requiring small amount of GPU's device memory. More information regarding this SpGEMM algorithm can be found in (2).

(1) Yusuke Nagasaka, Akira Nukada, Satoshi Matsuoka, "Adaptive Multi-level Blocking Optimization for Sparse Matrix Vector Multiplication on GPU", International Conference on Computational Science (ICCS 2016), June 2016

(2) Yusuke Nagasaka, Akira Nukada, Satoshi Matsuoka, "High-performance and Memory-saving Sparse General Matrix-Matrix Multiplication for NVIDIA Pascal GPU", International Conference on Parallel Processing (ICPP 2017), August 2017

## Requirement
- CUDA: >=5.0 && < 9.0 (assume implicit synchronization among WARP)
- Compute capability: >=3.5
- Thrust: should be installed for sort and scan operations  

## Preparation
To use this library, the first thing you need to do is to modify the Makefile with correct CUDA installation path. The compute capability is also appropriately set.

## Components
### C version
inc --- header files  
src --- 'conversion', 'kernel' and 'nsparse.cu' are library program. Sample codes are located in 'sample' folder.  
bin --- Execution files are generated.  

### C++ version (only SpMV)
inc --- header files  
sample --- sample codes  
bin --- Execution files are generated.  

## Execution of SpMV
Sample SpmV program executes Ax=y, where A is sparse matrix, x and y are dense vectors. The sample provides two SpMV program: one is with cuSPARSE library and the other is with AMB format. The command 'make' generates two executable files in 'bin' folder, 'amb_s' for single precision and 'amb_d' for double precision. The matrix data (in matrix market format) is indicated in first argument.

(1) run SpMV code on matrix data with auto-tuning in single precision  
./bin/amb_s ../data/test.mtx

## Execution of SpGEMM
Sample SpGEMM program executes C=A^2, where A and C are sparse matrices. The sample provides two SpGEMM program: one is with cuSPARSE library and the other is our hash-based SpGEMM algorithm. The command 'make spgemm_hash' generates two executable files in 'bin' folder, 'spgemm_hash_s' for single precision and 'spgemm_hash_d' for double precision. The matrix data (in matrix market format) is indicated in first argument.
./bin/spgemm_hash_s ../data/test.mtx

This library also provides the kernel generator to set appropriate parameter for your GPUs. Open 'src/kernel/spgemm_hash_kernel_gen.c', and check marked lines. Set hardware specific parameters such as shared memory size. After then, compile them and run.
make spgemm_hash_kernel_gen
This command generates the kernel programs for both single and double precision.

## To use with your own program
### C version
Copy 'inc' and 'src' except 'sample' folder to your codes. To use the library, include files in your code and compile with the program in 'src' with -DFLOAT (for single) or -DDOUBLE (for double) option (default is double precision).

### C++ version
Copy 'inc' folder to your codes. To use the library, include files in your code and compile the program with -DFLOAT (for single) or -DDOUBLE (for double) option (default is double precision).