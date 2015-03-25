#include	<wb.h>

#define BLOCK_SIZE 256
#define SEG_SIZE 256

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
	//@@ Insert code to implement vector addition here
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) {
		out[i] = in1[i] + in2[i];
	}
}

int main(int argc, char ** argv) {
	wbArg_t args;
	int inputLength;
	float * hostInput1;
	float * hostInput2;
	float * hostOutput;
	float * d_A0;
	float * d_B0;
	float * d_C0;
	float * d_A1;
	float * d_B1;
	float * d_C1;

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
	hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
	hostOutput = (float *)malloc(inputLength * sizeof(float));
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbTime_start(GPU, "Allocating GPU memory.");
	cudaMalloc((void **)&d_A0, SEG_SIZE*sizeof(float));
	cudaMalloc((void **)&d_B0, SEG_SIZE*sizeof(float));
	cudaMalloc((void **)&d_C0, SEG_SIZE*sizeof(float));
	cudaMalloc((void **)&d_A1, SEG_SIZE*sizeof(float));
	cudaMalloc((void **)&d_B1, SEG_SIZE*sizeof(float));
	cudaMalloc((void **)&d_C1, SEG_SIZE*sizeof(float));
	cudaStream_t stream0, stream1, stream2;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Performing CUDA computation.");
	for (int i = 0; i < inputLength; i += SEG_SIZE*2) {
		cudaMemcpyAsync(d_A0, &hostInput1[i], SEG_SIZE*sizeof(float), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(d_B0, &hostInput2[i], SEG_SIZE*sizeof(float), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(d_A1, &hostInput1[i + SEG_SIZE], SEG_SIZE*sizeof(float), cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(d_B1, &hostInput2[i + SEG_SIZE], SEG_SIZE*sizeof(float), cudaMemcpyHostToDevice, stream1);

		dim3 dimGrid((SEG_SIZE - 1) / BLOCK_SIZE + 1, 1, 1);
		dim3 dimBlock(BLOCK_SIZE, 1, 1);
		vecAdd <<<dimGrid, dimBlock, 0, stream0 >>>(d_A0, d_B0, d_C0, SEG_SIZE);
		vecAdd <<<dimGrid, dimBlock, 0, stream1 >>>(d_A1, d_B1, d_C1, SEG_SIZE);

		cudaMemcpyAsync(&hostOutput[i], d_C0, SEG_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(&hostOutput[i + SEG_SIZE], d_C1, SEG_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream1);

		cudaDeviceSynchronize();
	}
	wbTime_stop(GPU, "Performing CUDA computation.");

	wbSolution(args, hostOutput, inputLength);

	free(hostInput1);
	free(hostInput2);
	free(hostOutput);

	return 0;
}