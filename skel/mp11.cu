// Histogram Equalization

#include    <wb.h>

#define HISTOGRAM_LENGTH 256

#define BLOCK_WIDTH 16

//@@ insert code here
__device__ unsigned char correct_color(unsigned char val, float * cdf, float * cdfmin) {
	return (unsigned char) min(max(255.0 * (cdf[val] - *cdfmin) / (1.0 - *cdfmin), 0.0), 255.0);
}

__global__ void equalizeImage(unsigned char * image, float * cdf, float * cdfmin, int height, int width, int channels) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	int row = blockIdx.y*blockDim.y + ty;
	int col = blockIdx.x*blockDim.x + tx;
	int ch = tz;

	if (row < height && col < width) {
		int idx = (row*width + col)*channels + ch;
		image[idx] = correct_color(image[idx], cdf, cdfmin);
	}
}

__global__ void getMin(float * input, float * output, int len) {
	__shared__ float partial_min[HISTOGRAM_LENGTH];

	unsigned int t = threadIdx.x;
	unsigned int start = 2 * blockIdx.x * blockDim.x;

	// Load input into shared memory
	if (start + t < len) {
		partial_min[t] = input[start + t];
	}
	else {
		partial_min[t] = 0;
	}

	if (start + blockDim.x + t < len) {
		partial_min[t + blockDim.x] = input[start + blockDim.x + t];
	}
	else {
		partial_min[t + blockDim.x] = 0;
	}

	for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
		__syncthreads();
		if (t < stride) {
			partial_min[t] = min(partial_min[t], partial_min[t + stride]);
		}
	}

	output[blockIdx.x] = partial_min[0];
}

__global__ void getCDF(unsigned int * histogram, float * output, float * sum, int len, int height, int width) {
	__shared__ float scan_array[HISTOGRAM_LENGTH];

	unsigned int t = threadIdx.x;
	unsigned int start = 2 * blockIdx.x * blockDim.x;

	// Load input into shared memory
	if (start + t < len) {
		scan_array[t] = (float)histogram[start + t] / (float)(width*height);
	}
	else {
		scan_array[t] = 0.0;
	}

	if (start + blockDim.x + t < len) {
		scan_array[blockDim.x + t] = (float)histogram[start + blockDim.x + t] / (float)(width*height);
	}
	else {
		scan_array[blockDim.x + t] = 0.0;
	}

	__syncthreads();
	// Reduction
	for (int stride = 1; stride <= blockDim.x; stride *= 2) {
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

__global__ void float2char(float* input, unsigned char* output, int height, int width, int channels) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	int row = blockIdx.y*blockDim.y + ty;
	int col = blockIdx.x*blockDim.x + tx;
	int ch = tz;

	if (row < height && col < width) {
		output[(row*width + col)*channels + ch] = (unsigned char)(255 * input[(row*width + col)*channels + ch]);
	}
}

__global__ void char2float(unsigned char* input, float* output, int height, int width, int channels) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	int row = blockIdx.y*blockDim.y + ty;
	int col = blockIdx.x*blockDim.x + tx;
	int ch = tz;

	if (row < height && col < width) {
		output[(row*width + col)*channels + ch] = (float)(input[(row*width + col)*channels + ch] / 255.0);
	}
}

__global__ void rgb2gray(unsigned char* rgbinput, unsigned char* grayoutput, int height, int width, int channels) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = blockIdx.y*blockDim.y + ty;
	int col = blockIdx.x*blockDim.x + tx;

	if (row < height && col < width) {
		int idx = row*width + col;
		float r = rgbinput[3 * idx];
		float g = rgbinput[3 * idx + 1];
		float b = rgbinput[3 * idx + 2];
		grayoutput[idx] = (unsigned char)(0.21*r + 0.71*g + 0.07*b);
	}
}

__global__ void getHistogram(unsigned char* grayinput, int height, int width, unsigned int* histogram) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = blockIdx.y*blockDim.y + ty;
	int col = blockIdx.x*blockDim.x + tx;

	// Initialize private histogram
	int idx = ty*blockDim.x + tx;
	__shared__ unsigned int histo_private[HISTOGRAM_LENGTH];
	if (idx < HISTOGRAM_LENGTH) {
		histo_private[idx] = 0;
	}
	__syncthreads();

	// Build private histogram
	if (row < height && col < width) {
		atomicAdd(histo_private+grayinput[row*width + col], 1);
	}
	__syncthreads();

	
	if (idx < HISTOGRAM_LENGTH) {
		atomicAdd(histogram+idx, histo_private[idx]);
	}
}

int main(int argc, char ** argv) {
	wbArg_t args;
	int imageWidth;
	int imageHeight;
	int imageChannels;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float * hostInputImageData;
	float * hostOutputImageData;
	float * deviceInputImageData;
	unsigned char * deviceCharImageData;
	unsigned char * deviceGrayImageData;
	unsigned int * deviceHistogram;
	float * deviceCDF;
	float * deviceCDFmin;
	float * deviceOutputImageData;
	const char * inputImageFile;

	//@@ Insert more code here

	args = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(args, 0);

	wbTime_start(Generic, "Importing data and creating memory on host");
	inputImage = wbImport(inputImageFile);
	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);
	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceCharImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceGrayImageData, imageWidth * imageHeight * sizeof(float));
	cudaMalloc((void **)&deviceHistogram, HISTOGRAM_LENGTH * sizeof(int));
	cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH * sizeof(float));
	cudaMalloc((void **)&deviceCDFmin, sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");
	cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(int));
	cudaMemset(deviceCDF, 0.0, HISTOGRAM_LENGTH * sizeof(float));

	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputImageData,
		hostInputImageData,
		imageWidth * imageHeight * imageChannels * sizeof(float),
		cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");

	//@@ insert code here
	dim3 dimGrid((imageWidth - 1) / BLOCK_WIDTH + 1, (imageHeight - 1) / BLOCK_WIDTH + 1, 1);
	dim3 dimBlock3(BLOCK_WIDTH, BLOCK_WIDTH, imageChannels);
	dim3 dimBlock1(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	// Cast to uchar
	float2char << <dimGrid, dimBlock3 >> >(deviceInputImageData, deviceCharImageData, imageHeight, imageWidth, imageChannels);
	cudaDeviceSynchronize();
	// Convert to gray
	rgb2gray << <dimGrid, dimBlock1 >> >(deviceCharImageData, deviceGrayImageData, imageHeight, imageWidth, imageChannels);
	cudaDeviceSynchronize();
	// Calculate histogram
	getHistogram << <dimGrid, dimBlock1 >> >(deviceGrayImageData, imageHeight, imageWidth, deviceHistogram);
	cudaDeviceSynchronize();

	// Get CDF
	dim3 dimGrid1(1, 1, 1);
	dim3 dimBlock128(128, 1, 1);
	getCDF << <dimGrid1, dimBlock128 >> >(deviceHistogram, deviceCDF, NULL, HISTOGRAM_LENGTH, imageHeight, imageWidth);
	cudaDeviceSynchronize();
	// Get Min
	getMin <<<dimGrid1, dimBlock128 >>>(deviceCDF, deviceCDFmin, HISTOGRAM_LENGTH);
	cudaDeviceSynchronize();
	// Equalize
	equalizeImage << <dimGrid, dimBlock3 >> >(deviceCharImageData, deviceCDF, deviceCDFmin, imageHeight, imageWidth, imageChannels);
	cudaDeviceSynchronize();
	// Cast to float
	char2float <<<dimGrid, dimBlock3 >>>(deviceCharImageData, deviceOutputImageData, imageHeight, imageWidth, imageChannels);
	cudaDeviceSynchronize();

	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputImageData,
		deviceOutputImageData,
		imageWidth * imageHeight * imageChannels * sizeof(float),
		cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbSolution(args, outputImage);

	//@@ insert code here
	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);

	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	return 0;
}