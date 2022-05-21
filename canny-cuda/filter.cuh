#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void filterKernelV1(unsigned char *in_pixels, float *in_kernels, int N, int M, int h, int w, unsigned char *out_pixels, int n, int m) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < n && col < m) {
		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {
				out_pixels[row * m + col] += in_kernels[i * w + j] * in_pixels[(row + i) * M + col + j];
			}
		}
	}
}

__global__ void filterKernelV2(unsigned char *in_pixels, float *in_kernels, int N, int M, int h, int w, unsigned char *out_pixels, int n, int m) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float res = 0;
	if (row < n && col < m) {
		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {
				res += in_kernels[i * w + j] * in_pixels[(row + i) * M + col + j];
			}
		}
		out_pixels[row * m + col] = res;
	}
}

__global__ void filterKernelV3(unsigned char *in_pixels, float *in_kernels, int N, int M, int h, int w, unsigned char *out_pixels, int n, int m) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = threadIdx.y;
	int idx = threadIdx.x;

	__shared__ float data[32 * 2][32 * 2];
	data[idy * 2][idx * 2] = in_pixels[(row + idy) * M + col + idx];
	data[idy * 2 + 1][idx * 2] = in_pixels[(row + idy + 1) * M + col + idx];
	data[idy * 2][idx * 2 + 1] = in_pixels[(row + idy) * M + (col + idx + 1)];
	data[idy * 2 + 1][idx * 2 + 1] = in_pixels[(row + idy + 1) * M + (col + idx + 1)];
	__syncthreads();

	float res = 0;
	if (row < n && col < m) {
		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {
				res += in_kernels[i * w + j] * data[idy + i][idx + j];
			}
		}
		out_pixels[row * m + col] = res;
	}
}

__constant__ float kernel_dxy[13 * 13];
__global__ void filterKernelV4(unsigned char *in_pixels, int N, int M, int h, int w, unsigned char *out_pixels, int n, int m) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = threadIdx.y;
	int idx = threadIdx.x;

	__shared__ float data[32 * 2][32 * 2];
	data[idy * 2][idx * 2] = in_pixels[(row + idy) * M + col + idx];
	data[idy * 2 + 1][idx * 2] = in_pixels[(row + idy + 1) * M + col + idx];
	data[idy * 2][idx * 2 + 1] = in_pixels[(row + idy) * M + (col + idx + 1)];
	data[idy * 2 + 1][idx * 2 + 1] = in_pixels[(row + idy + 1) * M + (col + idx + 1)];
	__syncthreads();

	float res = 0;
	if (row < n && col < m) {
		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {
				res += kernel_dxy[i * w + j] * data[idy + i][idx + j];
			}
		}
		out_pixels[row * m + col] = res;
	}
}

__constant__ float kernel_dx[13];
__global__ void filterKernelV5Col(unsigned char *in_pixels, int N, int M, int k, unsigned char *out_pixels, int n, int m) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = threadIdx.y;
	int idx = threadIdx.x;

	__shared__ float data[32][32 * 2];
	data[idy][idx * 2] = in_pixels[row * M + col + idx];
	data[idy][idx * 2 + 1] = in_pixels[row * M + col + idx + 1];
	__syncthreads();

	float res = 0;
	if (row < n && col < m) {
		for (int j = 0; j < k; ++j) {
			res += kernel_dx[j] * data[idy][idx + j];
		}
		out_pixels[row * m + col] = res;
	}
}

__global__ void filterKernelV5Row(unsigned char *in_pixels, int N, int M, int k, unsigned char *out_pixels, int n, int m) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = threadIdx.y;
	int idx = threadIdx.x;

	__shared__ float data[32 * 2][32];
	data[idy * 2][idx] = in_pixels[(row + idy) * M + col];
	data[idy * 2 + 1][idx] = in_pixels[(row + idy + 1) * M + col];
	__syncthreads();

	float res = 0;
	if (row < n && col < m) {
		for (int i = 0; i < k; ++i) {
			res += kernel_dx[i] * data[idy + i][idx];
		}
		out_pixels[row * m + col] = res;
	}
}
