#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void SobelKernelV1(unsigned char *in_pixels, float *in_kernels, int N, int M, int h, int w, float *out_pixels, int n, int m) {

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

__global__ void SobelKernelV2(unsigned char *in_pixels, float *in_kernels, int N, int M, int h, int w, float *out_pixels, int n, int m) {

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

__global__ void SobelKernelV3Y(unsigned char *in_pixels, int N, int M, float *out_pixels, int n, int m) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float res = 0;
	int idx;
	if (row < n && col < m) {
		idx = row * M + col;
		res += -1 * in_pixels[idx];
		res += -2 * in_pixels[idx + 1];
		res += -1 * in_pixels[idx + 2];
		res += 1 * in_pixels[idx + 2 * M];
		res += 2 * in_pixels[idx + 2 * M + 1];
		res += 1 * in_pixels[idx + 2 * M + 2];
		out_pixels[row * m + col] = res;
	}
}

__global__ void SobelKernelV3X(unsigned char *in_pixels, int N, int M, float *out_pixels, int n, int m) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float res = 0;
	int idx;
	if (row < n && col < m) {
		idx = row * M + col;
		res += -1 * in_pixels[idx];
		res += 1 * in_pixels[idx + 2];
		res += -2 * in_pixels[idx + M];
		res += 2 * in_pixels[idx + M + 2];
		res += -1 * in_pixels[idx + 2 * M];
		res += 1 * in_pixels[idx + 2 * M + 2];
		out_pixels[row * m + col] = res;
	}
}

__global__ void SobelKernelV4Y(unsigned char *in_pixels, int N, int M, float *out_pixels, int n, int m) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

	float res = 0;
	int idx;
	for (int c = col; c < col + 4; c++) {
		if (row < n && c < m) {
			res = 0;
			idx = row * M + c;
			res += -1 * in_pixels[idx];
			res += -2 * in_pixels[idx + 1];
			res += -1 * in_pixels[idx + 2];
			res += 1 * in_pixels[idx + 2 * M];
			res += 2 * in_pixels[idx + 2 * M + 1];
			res += 1 * in_pixels[idx + 2 * M + 2];
			out_pixels[row * m + c] = res;
		}
	}
}

__global__ void SobelKernelV4X(unsigned char *in_pixels, int N, int M, float *out_pixels, int n, int m) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

	float res = 0;
	int idx;
	for (int c = col; c < col + 4; c++) {
		if (row < n && c < m) {
			res = 0;
			idx = row * M + c;
			res += -1 * in_pixels[idx];
			res += 1 * in_pixels[idx + 2];
			res += -2 * in_pixels[idx + M];
			res += 2 * in_pixels[idx + M + 2];
			res += -1 * in_pixels[idx + 2 * M];
			res += 1 * in_pixels[idx + 2 * M + 2];
			out_pixels[row * m + c] = res;
		}
	}
}

__global__ void SobelKernelV5Y(unsigned char *in_pixels, int N, int M, float *out_pixels, int n, int m) {
	int row = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float res = 0;
	int idx;
	for (int r = row; r < row + 4; r++) {
		if (r < n && col < m) {
			res = 0;
			idx = r * M + col;
			res += -1 * in_pixels[idx];
			res += -2 * in_pixels[idx + 1];
			res += -1 * in_pixels[idx + 2];
			res += 1 * in_pixels[idx + 2 * M];
			res += 2 * in_pixels[idx + 2 * M + 1];
			res += 1 * in_pixels[idx + 2 * M + 2];

			out_pixels[r * m + col] = res;
		}
	}
}

__global__ void SobelKernelV5X(unsigned char *in_pixels, int N, int M, float *out_pixels, int n, int m) {
	int row = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float res = 0;
	int idx;
	for (int r = row; r < row + 4; r++) {
		if (r < n && col < m) {
			res = 0;
			idx = r * M + col;
			res += -1 * in_pixels[idx];
			res += 1 * in_pixels[idx + 2];
			res += -2 * in_pixels[idx + M];
			res += 2 * in_pixels[idx + M + 2];
			res += -1 * in_pixels[idx + 2 * M];
			res += 1 * in_pixels[idx + 2 * M + 2];
			out_pixels[r * m + col] = res;
		}
	}
}

__global__ void SobelKernelV6Y(unsigned char *in_pixels, int N, int M, float *out_pixels, int n, int m) {
	int row = blockIdx.y * blockDim.y * 4;
	int col = blockIdx.x * blockDim.x;
	int idy = threadIdx.y * 4;
	int idx = threadIdx.x;
	
	__shared__ float data[32 + 2][32 + 2];

	for (int r = idy; r < idy + 4; r++) {
		if (r + row < N && col + idx < M) {
			data[r][idx] = in_pixels[(row + r) * M + col + idx];
		}
		if (idx == 31) {
			data[r][32] = in_pixels[(row + r) * M + col + 32];
			data[r][33] = in_pixels[(row + r) * M + col + 33];
		}
	}
	if (idy == 28) {
		data[32][idx] = in_pixels[(row + 32) * M + col + idx];
		data[33][idx] = in_pixels[(row + 33) * M + col + idx];
		if (idx == 31) {
			data[32][32] = in_pixels[(row + 32) * M + col + 32];
			data[32][33] = in_pixels[(row + 32) * M + col + 33];
			data[33][32] = in_pixels[(row + 33) * M + col + 32];
			data[33][33] = in_pixels[(row + 33) * M + col + 33];
		}
	}
	
	__syncthreads();
	

	float res = 0;
	for (int r = idy; r < idy + 4; r++) {
		if (r + row < n && col + idx < m) {
			res = 0;
			res += -1 * data[r][idx];
			res += -2 * data[r][idx + 1];
			res += -1 * data[r][idx + 2];
			res += 1 * data[r + 2][idx];
			res += 2 * data[r + 2][idx + 1];
			res += 1 * data[r + 2][idx + 2];
			out_pixels[(r + row) * m + col + idx] = res;
		}
	}
}

__global__ void SobelKernelV6X(unsigned char *in_pixels, int N, int M, float *out_pixels, int n, int m) {
	int row = blockIdx.y * blockDim.y * 4;
	int col = blockIdx.x * blockDim.x;
	int idy = threadIdx.y * 4;
	int idx = threadIdx.x;

	__shared__ float data[32 + 2][32 + 2];

	for (int r = idy; r < idy + 4; r++) {
		if (r + row < N && col + idx < M) {
			data[r][idx] = in_pixels[(row + r) * M + col + idx];
		}
		if (idx == 31) {
			data[r][32] = in_pixels[(row + r) * M + col + 32];
			data[r][33] = in_pixels[(row + r) * M + col + 33];
		}
	}
	if (idy == 28) {
		data[32][idx] = in_pixels[(row + 32) * M + col + idx];
		data[33][idx] = in_pixels[(row + 33) * M + col + idx];
		if (idx == 31) {
			data[32][32] = in_pixels[(row + 32) * M + col + 32];
			data[32][33] = in_pixels[(row + 32) * M + col + 33];
			data[33][32] = in_pixels[(row + 33) * M + col + 32];
			data[33][33] = in_pixels[(row + 33) * M + col + 33];
		}
	}

	__syncthreads();


	float res = 0;
	for (int r = idy; r < idy + 4; r++) {
		if (r + row < n && col + idx < m) {
			res = 0;
			res += 1 * data[r][idx];
			res += -1 * data[r][idx + 2];
			res += 2 * data[r + 1][idx];
			res += -2 * data[r + 1][idx + 2];
			res += 1 * data[r + 2][idx];
			res += -1 * data[r + 2][idx + 2];
			out_pixels[(r + row) * m + col + idx] = res;
		}
	}
}
