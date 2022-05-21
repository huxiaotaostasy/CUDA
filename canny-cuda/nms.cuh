#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define PI_1_4 0.78539815f
#define PI_1_2 1.5707963f
#define PI_3_4 2.3561944f
#define PI 3.1415926f
#define PI_5_4 3.92699075f
#define PI_6_4 4.7123889f
#define PI_7_4 5.497787f
#define PI2 6.2831852f


__global__ void nonMaximumSuppressionV1(float *in_magnitude, float *in_angle, int N, int M, float* out_pixels) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	//梯度方向上的四个梯度值
	float d1, d2, d3, d4;
	// d1和d2、d3和d4间插值时d1、d3权重
	float weight;
	float t1, t2; // 插值后的梯度
	float mag, ang;

	if (row > 0 && col > 0 && row < N - 1 && col < M - 1) {
		int idx = row * M + col;

		mag = in_magnitude[idx], ang = in_angle[idx];
		if (mag <= 0) {
			return;
		}

		/*
		----------*----
		-	   -	  -
		-	   *	  -
		-	   -	  -
		----*----------
		*/
		if (ang >= PI_1_4 && ang <= PI_1_2) {
			d1 = in_magnitude[idx - M];
			d2 = in_magnitude[idx - M + 1];
			d3 = in_magnitude[idx + M];
			d4 = in_magnitude[idx + M - 1];
			weight = tan(PI_1_2 - ang);
		}
		/*
		----*----------
		-	   -	  -
		-	   *	  -
		-	   -	  -
		----------*----
		*/
		else if (ang >= PI_1_2 && ang <= PI_3_4) {
			d1 = in_magnitude[idx - M];
			d2 = in_magnitude[idx - M - 1];
			d3 = in_magnitude[idx + M];
			d4 = in_magnitude[idx + M + 1];
			weight = tan(ang - PI_1_2);
		}
		/*
		---------------
		-	   -	  *
		-	   *	  -
		*	   -	  -
		---------------
		*/
		else if (ang >= 0 && ang <= PI_1_4) {
			d1 = in_magnitude[idx + 1];
			d2 = in_magnitude[idx - M + 1];
			d3 = in_magnitude[idx - 1];
			d4 = in_magnitude[idx + M - 1];
			weight = tan(ang);
		}
		/*
		---------------
		*	   -	  -
		-	   *	  -
		-	   -	  *
		---------------
		*/
		else if (ang >= PI_3_4 && ang <= PI) {
			d1 = in_magnitude[idx + 1];
			d2 = in_magnitude[idx + M + 1];
			d3 = in_magnitude[idx - 1];
			d4 = in_magnitude[idx - M - 1];
			weight = tan(PI - ang);
		}
		//插值
		t1 = weight * d1 + (1 - weight) * d2;
		t2 = weight * d3 + (1 - weight) * d4;

		if (mag > t1 && mag > t2) {
			out_pixels[idx] = mag;
		}
	}
}

__global__ void nonMaximumSuppressionV2(float *in_magnitude, float *in_angle, int N, int M, float* out_pixels) {
	int row = (blockIdx.y * blockDim.y + threadIdx.y) * 8;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	//梯度方向上的四个梯度值
	float d1, d2, d3, d4;
	// d1和d2、d3和d4间插值时d1、d3权重
	float weight;
	float t1, t2; // 插值后的梯度
	float mag, ang;
	for (int r = row; r < row + 8; r++) {
		if (r > 0 && col > 0 && r < N - 1 && col < M - 1) {
			int idx = r * M + col;

			mag = in_magnitude[idx], ang = in_angle[idx];
			if (mag <= 0) {
				return;
			}

			/*
			----------*----
			-	   -	  -
			-	   *	  -
			-	   -	  -
			----*----------
			*/
			if (ang >= PI_1_4 && ang <= PI_1_2) {
				d1 = in_magnitude[idx - M];
				d2 = in_magnitude[idx - M + 1];
				d3 = in_magnitude[idx + M];
				d4 = in_magnitude[idx + M - 1];
				weight = tan(PI_1_2 - ang);
			}
			/*
			----*----------
			-	   -	  -
			-	   *	  -
			-	   -	  -
			----------*----
			*/
			else if (ang >= PI_1_2 && ang <= PI_3_4) {
				d1 = in_magnitude[idx - M];
				d2 = in_magnitude[idx - M - 1];
				d3 = in_magnitude[idx + M];
				d4 = in_magnitude[idx + M + 1];
				weight = tan(ang - PI_1_2);
			}
			/*
			---------------
			-	   -	  *
			-	   *	  -
			*	   -	  -
			---------------
			*/
			else if (ang >= 0 && ang <= PI_1_4) {
				d1 = in_magnitude[idx + 1];
				d2 = in_magnitude[idx - M + 1];
				d3 = in_magnitude[idx - 1];
				d4 = in_magnitude[idx + M - 1];
				weight = tan(ang);
			}
			/*
			---------------
			*	   -	  -
			-	   *	  -
			-	   -	  *
			---------------
			*/
			else if (ang >= PI_3_4 && ang <= PI) {
				d1 = in_magnitude[idx + 1];
				d2 = in_magnitude[idx + M + 1];
				d3 = in_magnitude[idx - 1];
				d4 = in_magnitude[idx - M - 1];
				weight = -tan(ang);
			}
			//插值
			t1 = weight * d1 + (1 - weight) * d2;
			t2 = weight * d3 + (1 - weight) * d4;

			if (mag > t1 && mag > t2) {
				out_pixels[idx] = mag;
			}
		}
	}
}

__global__ void nonMaximumSuppressionV3(float *in_magnitude, float *in_angle, int N, int M, float* out_pixels) {
	int row = (blockIdx.y * blockDim.y + threadIdx.y) * 8;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	//梯度方向上的四个梯度值
	float d1, d2, d3, d4;
	// d1和d2、d3和d4间插值时d1、d3权重
	float weight;
	float t1, t2; // 插值后的梯度
	float mag, ang, theta;
	int n1, n2;
	for (int r = row; r < row + 8; r++) {
		if (r > 0 && col > 0 && r < N - 1 && col < M - 1) {
			int idx = r * M + col;

			mag = in_magnitude[idx], ang = in_angle[idx];
			if (mag <= 0) {
				return;
			}
			n1 = ((ang > PI_1_4) & (ang < PI_3_4)) * (-M - 1) + 1;
			n2 = (ang > PI_3_4) * (M + 1) + (ang > PI_1_2) * (-2) + (ang > PI_1_4) * (1 + M) - M;
			theta = (ang > PI_3_4) * (PI_1_2 - ang - ang) + (ang > PI_1_2) * (ang + ang) + (ang > PI_1_4) * (PI_1_2 - ang - ang) + ang;
			d1 = in_magnitude[idx + n1];
			d2 = in_magnitude[idx + n1 + n2];
			d3 = in_magnitude[idx - n1];
			d4 = in_magnitude[idx - n1 - n2];
			weight = tan(theta);
			
			//插值
			t1 = weight * d1 + (1 - weight) * d2;
			t2 = weight * d3 + (1 - weight) * d4;

			if (mag > t1 && mag > t2) {
				out_pixels[idx] = mag;
			}
		}
	}
}



