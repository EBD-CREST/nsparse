nsparse: Fast Sparse Matrix Library for GPU
======

## Author
Yusuke Nagasaka  
Akira Nukada  
Satoshi Matsuoka  

## Versions
1.0 (June, 2017)  


## Introduction
Sparse matrix computation is a key kernel of many applications. This library provides first sparse matrix computation kernels including SpMV and SpGEMM (will be released). A new sparse matrix format called the Adaptive Multi-level Blocking (AMB) format improves the performance of sparse matrix vector multiplication (SpMV). More information regarding AMB format can be found in (1).

(1) Yusuke Nagasaka, Akira Nukada, Satoshi Matsuoka, "Adaptive Multi-level Blocking Optimization for Sparse Matrix Vector Multiplication on GPU", International Conference on Computational Science (ICCS 2016), June 2016


#Requirement
- CUDA: >=5.0  
- Compute capability: >=3.5  
- Thrust: should be installed for sort and scan operations  


## Preparation
To use this library, the first thing you need to do is to modify the Makefile with correct CUDA instllation path. The compute capability is also appropriately set.


## Components
inc --- header files  
src --- 'conversion', 'kernel' and 'nsparse.cu' are library program. Sample codes are located in 'sample' folder.  
bin --- Execution files are generated.  


## Execution
Sample SpmV program executes Ax=y, where A is sparse matrix, x and y are dense vectors. The sample provides two SpMV program: one is with cuSPARSE library and the other is with AMB format. The command 'make' generates two executable files in 'bin' folder, 'amb_s' for single precision and 'amb_d' for double precision. The matrix data (in matrix market format) is indicated in first argument.

(1) run SpMV code on matrix data with auto-tuning in single precision  
./bin/amb_s ../data/test.mtx


## To use with your own program
Copy 'inc' and 'src' exept 'sample' folder to your codes. To use the library, include files in your code and compile with the program in 'src' with -DFLOAT (for single) or -DDOUBLE (for double) option (default is double precision).


