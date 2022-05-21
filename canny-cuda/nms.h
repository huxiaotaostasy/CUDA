#include<opencv2\opencv.hpp>
using namespace cv;

const float Pi = 3.14159265358979323846264338328;

#define PI_1_4 0.78539815f
#define PI_1_2 1.5707963f
#define PI_3_4 2.3561944f
#define PI 3.1415926f
#define PI_5_4 3.92699075f
#define PI_6_4 4.7123889f
#define PI_7_4 5.497787f
#define PI2 6.2831852f

void nonMaximumSuppression(const Mat& src, const Mat& angle, Mat& dst) {
	const int H = src.rows, W = src.cols;
	dst = Mat::zeros(H, W, CV_32F);

	//梯度方向上的四个梯度值
	float d1, d2, d3, d4;
	// d1和d2、d3和d4间插值时d1、d3权重
	float weight;
	float t1, t2; // 插值后的梯度
	for (int x = 1; x < H - 1; x++) {
		const float* mag = src.ptr<float>(x);
		const float* magTop = src.ptr<float>(x - 1);
		const float* magDown = src.ptr<float>(x + 1);
		const float* ang = angle.ptr<float>(x);
		float* data = dst.ptr<float>(x);
		for (int y = 1; y < W - 1; y++) {
			if (mag[y] > 0) {
				/*
				----------*----
				-	   -	  -
				-	   *	  -
				-	   -	  -
				----*----------
				*/
				if ((ang[y] >= 45 && ang[y] <= 90)||
					(ang[y] >= 225 && ang[y] <= 270)) {
					d1 = magTop[y];
					d2 = magTop[y + 1];
					d3 = magDown[y];
					d4 = magDown[y - 1];
					weight = tan((90 - ang[y])/180 * Pi);
				}
				/*
				----*----------
				-	   -	  -
				-	   *	  -
				-	   -	  -
				----------*----
				*/
				else if ((ang[y] >= 90 && ang[y] <= 135) ||
					(ang[y] >= 270 && ang[y] <= 315)) {
					d1 = magTop[y];
					d2 = magTop[y - 1];
					d3 = magDown[y];
					d4 = magDown[y + 1];
					weight = tan((ang[y] - 90) / 180 * Pi);
				}
				/*
				---------------
				-	   -	  *
				-	   *	  -
				*	   -	  -
				---------------
				*/
				else if ((ang[y] >= 0 && ang[y] <= 45) ||
					(ang[y] >= 180 && ang[y] <= 225)) {
					d1 = mag[y + 1];
					d2 = magTop[y + 1];
					d3 = mag[y - 1];
					d4 = magDown[y - 1];
					weight = tan(ang[y] / 180 * Pi);
				}
				/*
				---------------
				*	   -	  -
				-	   *	  -
				-	   -	  *
				---------------
				*/
				else if ((ang[y] >= 315 && ang[y] <= 360) ||
					(ang[y] >= 135 && ang[y] <= 180)) {
					d1 = mag[y - 1];
					d2 = magTop[y - 1];
					d3 = mag[y + 1];
					d4 = magDown[y + 1];
					weight = tan((180 - ang[y]) / 180 * Pi);
				}
				//插值
				t1 = weight * d1 + (1 - weight) * d2;
				t2 = weight * d3 + (1 - weight) * d4;
				if (mag[y] > t1 && mag[y] > t2) {
					data[y] = mag[y];
				}
			}
		}
	}
}

void nonMaximumSuppression2(const Mat& src, const Mat& angle, Mat& dst) {
	const int H = src.rows, W = src.cols;
	dst = Mat::zeros(H, W, CV_32F);

	//梯度方向上的四个梯度值
	float d1, d2, d3, d4;
	// d1和d2、d3和d4间插值时d1、d3权重
	float weight, theta, phase;
	float t1, t2; // 插值后的梯度
	int n1, n2;
	for (int x = 1; x < H - 1; x++) {
		const float* mag = src.ptr<float>(x);
		const float* ang = angle.ptr<float>(x);
		float* data = dst.ptr<float>(x);
		for (int y = 1; y < W - 1; y++) {
			if (mag[y] > 0) {
				phase = ang[y] <= 180 ? ang[y] : ang[y] - 180;
				phase = phase / 180 * PI;
				n1 = ((phase > PI_1_4) & (phase < PI_3_4)) * (-W - 1) + 1;
				n2 = (phase > PI_3_4) * (W + 1) + (phase > PI_1_2) * (-2) + (phase > PI_1_4) * (1 + W) - W;
				theta = (phase > PI_3_4) * (PI_1_2 - phase - phase) + (phase > PI_1_2) * (phase + phase) + (phase > PI_1_4) * (PI_1_2 - phase - phase) + phase;
				d1 = mag[y + n1];
				d2 = mag[y + n1 + n2];
				d3 = mag[y - n1];
				d4 = mag[y - n1 - n2];
				weight = tan(theta);

				//插值
				t1 = weight * d1 + (1 - weight) * d2;
				t2 = weight * d3 + (1 - weight) * d4;

				if (mag[y] > t1 && mag[y] > t2) {
					data[y] = mag[y];
				}
			}
		}
	}
}