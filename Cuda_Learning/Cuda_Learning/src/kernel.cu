////////////////////////////////////////////////////////////////!
//!                                                            	
//! File : kernel.cu
//!
//! Auther : Sharun Sashikumar
//!
//! Brief : Basic CUDA commands and Driver main method
//!
//! Date : Nov,DEC 2020
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
#include "Functions.h"
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

__global__ void Unique_Id_Cal_2D(int* Array) {
	int rowOffset = gridDim.x * blockDim.x * blockIdx.y;
	int blockOffset = blockIdx.x * blockDim.x;
	int thrdId = threadIdx.x;
	int index = thrdId + rowOffset + blockOffset;
	printf("Thread Id : %d Index : %d Value : %d\n", thrdId, index, Array[index]);

}

#endif 


#ifdef CUDA_MEM_TRNSFR

__global__ void MemTransferTest(int* mem) {

	int gId = threadIdx.x + (blockIdx.x * blockDim.x);
	printf("mem[%d] : %d\n",gId,mem[gId]);
}


#endif


#ifdef EXERCISE_2

__global__ void  PrintArray(int* mem) {

	//Print all the 64 elemets of the array
	int a = (threadIdx.x + (threadIdx.y * blockDim.y) + (threadIdx.z * blockDim.z * blockDim.y));//0 - 7
	int b = (blockDim.x * blockDim.y * blockDim.z); // 8

	int gid = a + ( blockIdx.x * b + (blockIdx.y * b ) * (gridDim.y)  +  (blockIdx.z * b * (gridDim.y) * (gridDim.z)) );
	
	printf("ARRAY[%d] : %d\n", gid,mem[gid]);


}

#endif

#ifdef VALIDITY_CHECK

__global__ void ValidityCheck(int* first, int* second, int* res, int size) {

	int gid = threadIdx.x + (blockIdx.x * blockDim.x); //Given is a 1-D thread block

	if (gid < size) {
	
		res[gid] = first[gid] + second[gid];
	}

}

#endif

#ifdef ERROR_HANDLE

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__,__LINE__);}

__global__ void MemTransferTest(int* mem) {

	int gId = threadIdx.x + (blockIdx.x * blockDim.x);
	printf("mem[%d] : %d\n", gId, mem[gId]);
}


inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {

	if (code != cudaSuccess) {
		fprintf(stderr, "GPU Assert : %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			return;

	}
}

#endif

#ifdef DEV_PROP

void QueryDeviceProperties() {

	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0) {
		printf("No CUDA device found\n");
	
	}

	int devNum = 0; //assuming only one cuda device
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, devNum);
	
	printf("Device %d : %s\n",devNum, devProp.name);
	printf("No of multiprocessors :				%d\n", devProp.multiProcessorCount);
	printf("Clock Rate : %d\n", devProp.clockRate);
	printf("Compute Capibitlity : %d.%d\n", devProp.major, devProp.minor);
	printf("Global memory count : %4.2f MB\n", devProp.totalGlobalMem / (1024.0 * 1024.0));
	printf("Constant  memory count : %4.2f MB\n", devProp.totalConstMem / (1024.0 * 1024.0));
	printf("Shared memory per block : %4.2f MB\n", devProp.sharedMemPerBlock / (1024.0 * 1024.0));


}
#endif

#ifdef BRANCH_EFFICIENCY

__global__ void branch_efficiency_Good() {

	int gid = blockIdx.x + blockDim.x + threadIdx.x;
	float a, b;
	a = b = 0;

	int wrapId = gid / 32;

	if (wrapId % 2 == 0) {
		a = 101.0;
		b = 200.0;
	
	}
	else {
		a = 10.0;
		b = 5.0;
	
	}
}


__global__ void branch_efficiency_Bad() {

	int gid = blockIdx.x + blockDim.x + threadIdx.x;
	float a, b;
	a = b = 0;

	if (gid % 2 == 0) {
		a = 101.0;
		b = 200.0;

	}
	else {
		a = 10.0;
		b = 5.0;

	}
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

	Unique_Id_Cal << <grid,block >> > (gpu_data); // 1 thread block with n threads
	cudaDeviceSynchronize();
	std::cout << "##################################" << std::endl;


	dim3 grid1(4, 1, 1);
	dim3 block1(2, 1, 1);
	Unique_Id_Auto_Cal << <grid1,block1 >> > (gpu_data); //2 thread in same dimension with in total n threads
	cudaDeviceSynchronize();
	std::cout << "##################################" << std::endl;


	dim3 grid2(2, 2, 1);
	dim3 block2(2, 1, 1);
	Unique_Id_Cal_2D << <grid2, block2 >> > (gpu_data); // 2 D arrangement
	
#endif

#ifdef CUDA_MEM_TRNSFR
	
	int memSize = 128;
	int byteSize = memSize * sizeof(int);

	int* host_mem_ptr = (int*)malloc(byteSize);

	//intialise the allocated memory
	for (int i = 0; i < memSize; ++i) {
		host_mem_ptr[i] = i + 12;
	}

	//allocating memory in gpu
	int* gpu_mem_ptr;
	cudaMalloc((void**)&gpu_mem_ptr, byteSize);

	//transfering memory from host to device (GPU)
	cudaMemcpy(gpu_mem_ptr, host_mem_ptr, byteSize, cudaMemcpyHostToDevice);

	dim3 grid(2, 1, 1);
	dim3 block(64, 1, 1);

	//invoking kernel
	MemTransferTest << <grid,block >> > (gpu_mem_ptr);
	
#endif

#ifdef EXERCISE_2

	//64 bytes array 
	const int m_size = 64;
	const int mem_bytes = m_size * sizeof(int);
	//printf("%d\n", mem_bytes);
	int* host_mem = (int*)malloc(mem_bytes);

	for (int i = 0; i < m_size; i++) {
		host_mem[i] = i + 100;
	} //allocated and initialized 64 elements in an array

	int* dev_mem = 0;
	cudaMalloc((void**)&dev_mem, mem_bytes);
	cudaMemcpy(dev_mem, host_mem,mem_bytes, cudaMemcpyHostToDevice);
	//allocate and set memory in GPU


	dim3 grid(2,2,2);
	dim3 block(2, 2, 2);

	PrintArray << < grid, block >> > (dev_mem);


	

#endif

#ifdef	VALIDITY_CHECK

	clock_t cpu_start = 0, cpu_end = 0 ,cpu_time= 0;
	clock_t gpu_start = 0, gpu_end = 0, gpu_t = 0;
	int items = 10000;
	int block_Size = 128;

	int SIZE = items * sizeof(items);

	int* h_first, * h_second, * h_res, *h_cmpRes = 0 ;
	int* gpu_first, * gpu_second, * gpu_res = 0;

	h_first = (int*)malloc(SIZE);
	h_second = (int*)malloc(SIZE);
	h_res = (int*)malloc(SIZE);
	h_cmpRes = (int*)malloc(SIZE);

	cpu_start = clock();
	init(h_first, items);
	init(h_second, items);
	
	SumArray(h_first, h_second, h_cmpRes, items);
 
	cpu_time = clock() - cpu_start;
	printf("CPU end : % f \n", clock() / CLOCKS_PER_SEC);

	//cudaMalloc((void**)&dev_mem, mem_bytes);
	gpu_start = clock();
	cudaMalloc((void**)&gpu_first, SIZE);
	cudaMalloc((void**)&gpu_second, SIZE);
	cudaMalloc((void**)&gpu_res, SIZE);
	
	cudaMemcpy(gpu_first, h_first, SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_second, h_second, SIZE, cudaMemcpyHostToDevice);
	gpu_end = clock();
	gpu_t = gpu_end - gpu_start;
	gpu_end = 0;
	gpu_start = 0;

	dim3 block(128, 1, 1);
	dim3 grid(items / block_Size + 1, 1, 1);
	gpu_start = clock();
	ValidityCheck << <block,grid >> > (gpu_first,gpu_second,gpu_res,items);

#endif

#ifdef ERROR_HANDLE

	int memSize = 128;
	int byteSize = memSize * sizeof(int);

	int* host_mem_ptr = (int*)malloc(byteSize);

	//intialise the allocated memory
	for (int i = 0; i < memSize; ++i) {
		host_mem_ptr[i] = i + 12;
	}

	//allocating memory in gpu
	int* gpu_mem_ptr;
	gpuErrorCheck(cudaMalloc((void**)&gpu_mem_ptr, byteSize));

	//transfering memory from host to device (GPU)
	gpuErrorCheck(cudaMemcpy(gpu_mem_ptr, host_mem_ptr, byteSize, cudaMemcpyHostToDevice));

	dim3 grid(2, 1, 1);
	dim3 block(64, 1, 1);

	//invoking kernel
	MemTransferTest << <grid, block >> > (gpu_mem_ptr);
#endif

#ifdef BRANCH_EFFICIENCY

	int size =  1 << 22;
	dim3 blk_siz(128);
	dim3 gri_Siz((size + blk_siz.x - 1) / blk_siz.x);

	branch_efficiency_Good << < blk_siz,gri_Siz>> > ();
	cudaDeviceSynchronize();
	branch_efficiency_Bad << < blk_siz, gri_Siz >> > ();

#endif
	cudaDeviceSynchronize();

#ifdef EXERCISE_2
	cudaFree(dev_mem);
	free(host_mem);
#endif
#ifdef CUDA_MEM_TRNSFR	
	cudaFree(gpu_mem_ptr);
	free(host_mem_ptr);
#endif

#ifdef VALIDITY_CHECK
	cudaMemcpy(h_res, gpu_res, SIZE, cudaMemcpyDeviceToHost);
	printf("CPU time : % u \nGPU time : % u\n", cpu_time, gpu_t);
	CheckValid(h_res, h_cmpRes,items);
	cudaFree(gpu_first);
	cudaFree(gpu_second);
	cudaFree(gpu_res);
	free(h_first);
	free(h_second);
	free(h_res);

#endif

#ifdef DEV_PROP
	QueryDeviceProperties();

#endif

	cudaDeviceReset();
	
	return 0;
}

//===============================================================
//
//===============================================================

//===============================================================
//                        END OF FILE
//===============================================================