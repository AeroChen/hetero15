// MP 5 Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                 \
		cudaError_t err = stmt;                            \
		if (err != cudaSuccess) {                          \
			wbLog(ERROR, "Failed to run stmt ", #stmt);    \
			return -1;                                     \
		}                                                  \
	} while(0)

__global__ void addSum(float * output, float * sum, int len) {
	__shared__ float prev_sum;

	unsigned int t = threadIdx.x;
	unsigned int start = 2 * blockIdx.x * blockDim.x;

	if (blockIdx.x > 0) {
		if (t == 0) prev_sum = sum[blockIdx.x - 1];
		__syncthreads();

		if (start + t < len) output[start + t] += prev_sum;
		if (start + blockDim.x + t < len) output[start + blockDim.x + t] += prev_sum;
	}
}

__global__ void scan(float * input, float * output, float * sum, int len) {
	//@@ Modify the body of this function to complete the functionality of
	//@@ the scan on the device
	//@@ You may need multiple kernel calls; write your kernels before this
	//@@ function and call them from here
	__shared__ float scan_array[2 * BLOCK_SIZE];

	unsigned int t = threadIdx.x;
	unsigned int start = 2 * blockIdx.x * blockDim.x;

	// Load input into shared memory
	if (start + t < len) {
		scan_array[t] = input[start + t];
	} else {
		scan_array[t] = 0.0;
	}

	if (start + blockDim.x + t < len) {
		scan_array[blockDim.x + t] = input[start + blockDim.x + t];
	} else {
		scan_array[blockDim.x + t] = 0.0;
	}
	
	__syncthreads();
	// Reduction
	for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
		int index = (t + 1) * stride * 2 - 1;
		if (index < 2 * blockDim.x) scan_array[index] += scan_array[index - stride];
		__syncthreads();
	}

	// Post reduction reverse
	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		int index = (t + 1) * stride * 2 - 1;
		if (index + stride < 2 * blockDim.x) scan_array[index + stride] += scan_array[index];
		__syncthreads();
	}
	
	if (start + t < len) output[start + t] = scan_array[t];
	if (start + blockDim.x + t < len) output[start + blockDim.x + t] = scan_array[blockDim.x + t];

	if (sum && t == 0) sum[blockIdx.x] = scan_array[2 * blockDim.x - 1];
}

int main(int argc, char ** argv) {
	wbArg_t args;
	float * hostInput; // The input 1D list
	float * hostOutput; // The output list
	float * deviceInput;
	float * deviceOutput;
	float * deviceSum;
	int numElements; // number of elements in the list

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
	hostOutput = (float*) malloc(numElements * sizeof(float));
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The number of input elements in the input is ", numElements);

	wbTime_start(GPU, "Allocating GPU memory.");
	wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
	wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Clearing output memory.");
	wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
	wbTime_stop(GPU, "Clearing output memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	//@@ Initialize the grid and block dimensions here
	int numBlocks = (numElements - 1) / (BLOCK_SIZE << 1) + 1;
	wbCheck(cudaMalloc((void **)&deviceSum, numBlocks*sizeof(float)));
	dim3 dimGrid(numBlocks, 1, 1);
	dim3 dimBlock(BLOCK_SIZE, 1, 1);

	wbTime_start(Compute, "Performing CUDA computation");
	//@@ Modify this to complete the functionality of the scan
	//@@ on the deivce
	scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, deviceSum, numElements);
	cudaDeviceSynchronize();
	scan<<<dim3(1,1,1), dimBlock>>>(deviceSum, deviceSum, NULL, numBlocks);
	cudaDeviceSynchronize();
	addSum<<<dimGrid, dimBlock>>>(deviceOutput, deviceSum, numElements);
	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	cudaFree(deviceInput);
	cudaFree(deviceOutput);
	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, hostOutput, numElements);

	free(hostInput);
	free(hostOutput);

	return 0;
}

