#include<opencv2\opencv.hpp>
using namespace cv;


void getGaussian(Mat& kernel, int k, float sigma) {
	CV_Assert(k % 2 == 1);
	kernel.create(k, k, CV_32F);
	Mat x(1, k, CV_32F);
	float* ptr = x.ptr<float>(0);
	float sigma2 = -0.5 / (pow(sigma, 2));
	float center = (k - 1) / 2;
	for (int i = 0; i < k; i++) {
		float offset = abs(i - center);
		ptr[i] = exp(offset * offset * sigma2);
	}
	x /= sum(x).val[0];
	kernel = x.t() * x;
}

void getGaussianSinge(Mat& kernel, int k, float sigma) {
	CV_Assert(k % 2 == 1);
	kernel.create(k, 1, CV_32F);
	float sigma2 = -0.5 / (pow(sigma, 2));
	float center = (k - 1) / 2;
	for (int i = 0; i < k; i++) {
		float offset = abs(i - center);
		kernel.at<float>(i, 0) = exp(offset * offset * sigma2);
	}
	kernel /= sum(kernel).val[0];
}

void getGaussianDerivative(Mat& kernelX, Mat& kernelY, int k, float sigma) {
	CV_Assert(k % 2 == 1);
	kernelX.create(k, k, CV_32F);
	kernelY.create(k, k, CV_32F);
	float sigma2 = -0.5 / (pow(sigma, 2));
	int w = k / 2;
	for (int x = -w; x <= w; x++) {
		for (int y = -w; y <= w; y++) {
			kernelX.at<float>(x + w, y + w) = - x * exp((x * x + y * y) * sigma2);
			kernelY.at<float>(x + w, y + w) = - y * exp((x * x + y * y) * sigma2);
		}
	}
}

void getSobel(Mat& kernelX, Mat& kernelY) {
	kernelX = (Mat_<float>(3, 3) << 1, 0, -1, 2, 0, -2, 1, 0, -1);
	kernelY = (Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
}

void filter(const Mat& src, Mat& dst, const Mat& kernel, int borderType = BORDER_DEFAULT, const Scalar& value = Scalar()) {
	const int H = src.rows, W = src.cols;
	const int h = kernel.rows, w = kernel.cols;

	Mat srcPadding;
	copyMakeBorder(src, srcPadding, h / 2, h / 2, w / 2, w / 2, borderType, value);

	dst = Mat::zeros(H, W, CV_32FC3);

	for (int x = 0; x < H; x++) {
		Vec3d* data = dst.ptr<Vec3d>(x);
		for (int y = 0; y < W; y++) {
			for (int i = 0; i < h; i++) {
				const Vec3b* p = srcPadding.ptr<Vec3b>(x + i);
				const float* k = kernel.ptr<float>(i);
				for (int j = 0; j < w; j++) {
					data[y][0] += p[y + j][0] * k[j];
					data[y][1] += p[y + j][1] * k[j];
					data[y][2] += p[y + j][2] * k[j];
				}
			}

		}
	}
	dst.convertTo(dst, CV_8UC3);
}

void filterGray(const Mat& src, Mat& dst, const Mat& kernel, bool convert2Byte=false, int borderType = BORDER_DEFAULT, const Scalar& value = Scalar()) {
	const int H = src.rows, W = src.cols;
	const int h = kernel.rows, w = kernel.cols;

	Mat srcPadding;
	copyMakeBorder(src, srcPadding, h / 2, h / 2, w / 2, w / 2, borderType, value);

	dst = Mat::zeros(H, W, CV_32FC1);

	for (int x = 0; x < H; x++) {
		float* data = dst.ptr<float>(x);
		for (int y = 0; y < W; y++) {
			for (int i = 0; i < h; i++) {
				const uchar* p = srcPadding.ptr<uchar>(x + i);
				const float* k = kernel.ptr<float>(i);
				for (int j = 0; j < w; j++) {
					data[y] += p[y + j] * k[j];
				}
			}
		}
	}

	if (convert2Byte) {
		dst.convertTo(dst, CV_8U);
	}
}

void filterGraySobelX(const Mat& src, Mat& dst, bool convert2Byte = false, int borderType = BORDER_DEFAULT, const Scalar& value = Scalar()) {
	const int H = src.rows, W = src.cols;

	Mat srcPadding;
	copyMakeBorder(src, srcPadding, 1, 1, 1, 1, borderType, value);

	dst = Mat::zeros(H, W, CV_32FC1);

	for (int x = 0; x < H; x++) {
		float* data = dst.ptr<float>(x);
		for (int y = 0; y < W; y++) {
			const uchar* t = srcPadding.ptr<uchar>(x);
			const uchar* c = srcPadding.ptr<uchar>(x + 1);
			const uchar* d = srcPadding.ptr<uchar>(x + 2);
			data[y] += -1 * t[y];
			data[y] += 1 * t[y + 2];
			data[y] += -2 * c[y];
			data[y] += 2 * c[y + 2];
			data[y] += -1 * d[y];
			data[y] += 1 * d[y + 2];
		}
	}

	if (convert2Byte) {
		dst.convertTo(dst, CV_8U);
	}
}

void filterGraySobelY(const Mat& src, Mat& dst, bool convert2Byte = false, int borderType = BORDER_DEFAULT, const Scalar& value = Scalar()) {
	const int H = src.rows, W = src.cols;

	Mat srcPadding;
	copyMakeBorder(src, srcPadding, 1, 1, 1, 1, borderType, value);

	dst = Mat::zeros(H, W, CV_32FC1);

	for (int x = 0; x < H; x++) {
		float* data = dst.ptr<float>(x);
		for (int y = 0; y < W; y++) {
			const uchar* t = srcPadding.ptr<uchar>(x);
			const uchar* d = srcPadding.ptr<uchar>(x + 2);
			data[y] += t[y];
			data[y] += 2 * t[y + 1];
			data[y] += 1 * t[y + 2];
			data[y] += -1 * d[y];
			data[y] += -2 * d[y + 1];
			data[y] += -1 * d[y + 2];
		}
	}

	if (convert2Byte) {
		dst.convertTo(dst, CV_8U);
	}
}