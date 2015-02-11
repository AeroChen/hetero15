#include    <wb.h>

#define wbCheck(stmt) do {                                 \
		cudaError_t err = stmt;                            \
		if (err != cudaSuccess) {                          \
			wbLog(ERROR, "Failed to run stmt ", #stmt);    \
			return -1;                                     \
		}                                                  \
	} while(0)


#define Mask_width  5
#define Mask_radius Mask_width/2
#define O_TILE_WIDTH 12
#define BLOCK_WIDTH O_TILE_WIDTH+Mask_width-1

//@@ INSERT CODE HERE
__global__ void convolution_2D(float* input, float* output, int height, int width, int channels, const float* __restrict__ mask) {
	__shared__ float ds_input[BLOCK_WIDTH][BLOCK_WIDTH][3];

	// Output index map
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	int row_o = blockIdx.y * O_TILE_WIDTH + ty;
	int col_o = blockIdx.x * O_TILE_WIDTH + tx;
	// Input index map
	int row_i = row_o - Mask_radius;
	int col_i = col_o - Mask_radius;

	if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width)) {
		ds_input[ty][tx][tz] = input[(row_i*width + col_i) * channels + tz];
	}
	else {
		ds_input[ty][tx][tz] = 0.0f;
	}
	__syncthreads();

	if (tx < O_TILE_WIDTH && ty < O_TILE_WIDTH) {
		float result = 0.0f;
		for (int i = 0; i < Mask_width; i++)
		{
			for (int j = 0; j < Mask_width; j++)
			{
				result += mask[i*Mask_width + j] * ds_input[ty + i][tx + j][tz];
			}
		}
		if (row_o < height && col_o < width) {
			output[(row_o*width + col_o) * channels + tz] = min(max(result, 0.0f), 1.0f);
		}
	}
}

int main(int argc, char* argv[]) {
	wbArg_t arg;
	int maskRows;
	int maskColumns;
	int imageChannels;
	int imageWidth;
	int imageHeight;
	char * inputImageFile;
	char * inputMaskFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float * hostInputImageData;
	float * hostOutputImageData;
	float * hostMaskData;
	float * deviceInputImageData;
	float * deviceOutputImageData;
	float * deviceMaskData;

	arg = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(arg, 0);
	inputMaskFile = wbArg_getInputFile(arg, 1);

	inputImage = wbImport(inputImageFile);
	hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

	assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
	assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);

	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");


	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputImageData,
			   hostInputImageData,
			   imageWidth * imageHeight * imageChannels * sizeof(float),
			   cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMaskData,
			   hostMaskData,
			   maskRows * maskColumns * sizeof(float),
			   cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");


	wbTime_start(Compute, "Doing the computation on the GPU");
	//@@ INSERT CODE HERE
	dim3 dimGrid((imageWidth - 1) / O_TILE_WIDTH + 1, (imageHeight - 1) / O_TILE_WIDTH + 1, 1);
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, imageChannels);
	convolution_2D<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceOutputImageData, imageHeight, imageWidth, imageChannels, deviceMaskData);
	wbTime_stop(Compute, "Doing the computation on the GPU");


	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputImageData,
			   deviceOutputImageData,
			   imageWidth * imageHeight * imageChannels * sizeof(float),
			   cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(arg, outputImage);

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);
	cudaFree(deviceMaskData);

	free(hostMaskData);
	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	return 0;
}
