////////////////////////////////////////////////////////////////!
//!                                                            	
//! File : kernel.cu
//!
//! Auther : Sharun Sashikumar
//!
//! Brief : Basic CUDA commands and Driver main method
//!
//! Date : Nov 2020
//!	
//!
////////////////////////////////////////////////////////////////!

//===============================================================

//===============================================================
//                        INCLUDES
//===============================================================
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Enable_Que.h"
#include <stdio.h>
#include <iostream>
#include <array>

//===============================================================
//
//===============================================================

//===============================================================
//                        TYPEDEFs
//===============================================================

//===============================================================
//                          DATA
//===============================================================

//===============================================================
//
//===============================================================


#ifdef HELLO_WORLD
//===============================================================
//!
//! brief : Device code for printing
//! 
//! args : --
//!
__global__ void HelloCUDA() {
	printf("Hello CUDA World\n");
}
#endif

#ifdef THREAD_ID
//===============================================================
//!
//! brief : Device code for printing threadId
//! 
//! args : --
//!
__global__ void PrintThreadId() {

	printf("threadIdx.x : %d threadIdx.y : %d threadIdx.y : %d\n",
	threadIdx.x,threadIdx.y,threadIdx.z);
	//Each thread will print its it as per its threadblock
}
#endif

#ifdef DETAILED_IDENT
//===============================================================
//!
//! brief : Device code for printing some of the Unique pre-defined identifiers in CUDA
//! 
//! args : --
//!
__global__ void PrintDetails() {

	printf("blockIdx.x : %d blockIdx.y : %d blockIdx.z : %d blockDim.x : %d blockDim.y : %d blockDim.z : %d gridDim.x : %d gridDim.y : %d gridDim.z : %d\n",
		blockIdx.x , blockIdx.y,blockIdx.z,blockDim.x,blockDim.y,blockDim.z,gridDim.x,gridDim.y,gridDim.z		
		);

}
#endif

#ifdef EXERCISE_1
__global__ void PrintExercise_1() {

	printf("threadIdx.x : %d threadIdx.y : %d threadIdx.z : %d blockIdx.x : %d blockIdx.y : %d blockIdx.z : %d gridDim.x : %d gridDim.y : %d gridDim.z : %d\n",
		threadIdx.x, threadIdx.y , threadIdx.z ,blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z
	);

}
#endif

#ifdef UNIQUE_INDEX
//===============================================================
//!
//! brief : Device code for provide a particular data to a thread
//! 
//! args : pointer to an array of integers
//!
__global__ void Unique_Id_Cal( int * Array ) {

	int thrdId = threadIdx.x;
	printf("Thread Id : %d Value : %d\n", thrdId, Array[thrdId]);
}

__global__ void Unique_Id_Auto_Cal(int* Array) {

	int Offset = blockIdx.x * blockDim.x;
	int thrdId = threadIdx.x;
	int index = thrdId + Offset;
	printf("Thread Id : %d Index : %d Value : %d\n", thrdId,index, Array[index]);
}

#endif 


//===============================================================
//!
//! brief : Main Driver method
//! 
//! args : --
//!
int main() {

#ifdef HELLO_WORLD
	//Version 1 of Launch Paramerters

	HelloCUDA << <1,1 >> > ();

	cudaDeviceSynchronize();
	std::cout << "##################################" << std::endl;
	//Version 2 of Launch Paramerters
	
	
	dim3 grid(8,2);
	dim3 block(2,2);
	HelloCUDA << <grid, block >> > ();


	cudaDeviceSynchronize();
	std::cout << "##################################" << std::endl;

	//Version 3 of Launch Paramerters
	int nx, ny;
	nx = 16;
	ny = 4;

	dim3 block1(8, 2);
	dim3 grid1(nx/block1.x, ny/block1.y);
	HelloCUDA << <grid1 , block1>> > ();


	cudaDeviceSynchronize();
	std::cout << "##################################" << std::endl;
#endif

#ifdef THREAD_ID
	//Looking into the threadIdx variable
	int nx = 16;
	int ny = 16;
	dim3 m_block(8, 8, 1);
	dim3 m_grid(nx / m_block.x, ny / m_block.y);
	PrintThreadId << <m_grid, m_block >> > ();
#endif

#ifdef  DETAILED_IDENT
	
	int nx = 4;
	int ny = 4;
	int nz = 4;
	dim3 m_block(2, 2, 2);
	dim3 m_grid(nx / m_block.x, ny / m_block.y, nz / m_block.z);
	PrintDetails << <m_grid, m_block >> > ();
#endif

#ifdef EXERCISE_1
	//Looking into other internal variables
	int nx = 4;
	int ny = 4;
	int nz = 4;
	dim3 m_block(2, 2, 2);
	dim3 m_grid(nx / m_block.x, ny / m_block.y,nz/m_block.z);
	//PrintDetails << <m_grid, m_block >> > ();
	PrintExercise_1 << < m_grid, m_block >> > ();
#endif


#ifdef UNIQUE_INDEX

	int host_Arr[] = { 23,4,2,2,4,5,564,34 };
	int size = sizeof(int) * ELEM;
	int* gpu_data;
	cudaMalloc((void **)&gpu_data,size);
	cudaMemcpy(gpu_data, host_Arr, size, cudaMemcpyHostToDevice);

	dim3 grid(1,1,1);
	dim3 block(8,1,1);

	Unique_Id_Cal << <grid,block >> > (gpu_data);
	cudaDeviceSynchronize();
	std::cout << "##################################" << std::endl;


	dim3 grid1(4, 1, 1);
	dim3 block1(2, 1, 1);

	Unique_Id_Auto_Cal << <grid1,block1 >> > (gpu_data);


#endif
	cudaDeviceSynchronize();

	cudaDeviceReset();

	return 0;
}

//===============================================================
//
//===============================================================

//===============================================================
//                        END OF FILE
//===============================================================