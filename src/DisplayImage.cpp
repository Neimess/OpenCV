#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace cv;

bool isPowerOfTwo(size_t n)
{
    if (n <= 0)
    {
        return false;
    }

    return (n & (n - 1)) == 0;
}

cv::Mat DFT_IMAGE(cv::Mat inputImage)
{
    int rows = inputImage.rows;
    int cols = inputImage.cols;
    cv::Mat dftReal(rows, cols, CV_32FC1, cv::Scalar(0.0, 0.0));
    cv::Mat dftImag(rows, cols, CV_32FC1, cv::Scalar(0.0, 0.0));

    for (int k = 0; k < rows; ++k)
    {
        for (int l = 0; l < cols; ++l)
        {
            for (int m = 0; m < rows; ++m)
            {
                for (int n = 0; n < cols; ++n)
                {
                    double angle = -2.0 * M_PI * ((static_cast<double>(m * k) / rows) + (static_cast<double>(n * l) / cols));

                    dftReal.at<float>(k, l) += inputImage.at<float>(m, n) * std::cos(angle);
                    dftImag.at<float>(k, l) += inputImage.at<float>(m, n) * std::sin(angle);
                }
            }
        }
    }
    cv::Mat dftMat;
    cv::merge(std::vector<cv::Mat>{dftReal, dftImag}, dftMat);
    return dftMat;
}

cv::Mat IDFT_IMAGE(cv::Mat dftMat)
{
    int rows = dftMat.rows;
    int cols = dftMat.cols;

    // Разделение вещественной и мнимой частей
    std::vector<cv::Mat> channels;
    cv::split(dftMat, channels);
    cv::Mat dftReal = channels[0];
    cv::Mat dftImag = channels[1];

    cv::Mat idftResult(rows, cols, CV_32FC1, cv::Scalar(0.0));

    for (int m = 0; m < rows; ++m)
    {
        for (int n = 0; n < cols; ++n)
        {
            float sumReal = 0.0;
            float sumImag = 0.0;

            for (int k = 0; k < rows; ++k)
            {
                for (int l = 0; l < cols; ++l)
                {
                    double angle = 2.0 * M_PI * ((static_cast<double>(m * k) / rows) + (static_cast<double>(n * l) / cols));
                    sumReal += dftReal.at<float>(k, l) * std::cos(angle) - dftImag.at<float>(k, l) * std::sin(angle);
                    sumImag += dftReal.at<float>(k, l) * std::sin(angle) + dftImag.at<float>(k, l) * std::cos(angle);
                }
            }

            idftResult.at<float>(m, n) = static_cast<float>((sumReal + sumImag) / (rows * cols));
        }
    }

    return idftResult;
}

std::vector<std::complex<double>> DFT(const std::vector<std::complex<double>> &vector)
{
    int N = vector.size();
    std::vector<std::complex<double>> result(N, {0.0, 0.0});

    for (int k = 0; k < N; ++k)
    {
        for (int n = 0; n < N; ++n)
        {
            double angle = -2.0 * M_PI * k * n / N;
            std::complex<double> complex_exp = {cos(angle), sin(angle)};
            result[k] += vector[n] * complex_exp;
        }
    }

    return result;
}

cv::Mat convertComplexVectorToMat(const std::vector<std::complex<double>> &complexVector, size_t _rows, size_t _cols)
{
    cv::Mat resultMat(_rows, _cols, CV_64FC2);

    for (int i = 0; i < _rows; ++i)
    {
        for (int j = 0; j < _cols; ++j)
        {
            resultMat.at<cv::Vec2d>(i, j)[0] = complexVector[i * _cols + j].real(); // реальная часть комплексного числа
            resultMat.at<cv::Vec2d>(i, j)[1] = complexVector[i * _cols + j].imag(); // мнимая часть комплексного числа
        }
    }
    return resultMat;
}

std::vector<std::complex<double>> IDFT(const std::vector<std::complex<double>> &vector)
{
    int N = vector.size();
    std::vector<std::complex<double>> result(N, {0.0, 0.0});

    for (int n = 0; n < N; ++n)
    {
        for (int k = 0; k < N; ++k)
        {
            double angle = 2.0 * M_PI * k * n / N;
            std::complex<double> complex_exp = {cos(angle), sin(angle)};
            result[n] += vector[k] * complex_exp;
        }
        result[n] /= N; // Нормализация
    }

    return result;
}

// Helper function to reverse bits in an integer
unsigned int reverseBits(unsigned int num, int log2n)
{
    unsigned int result = 0;
    for (int i = 0; i < log2n; ++i)
    {
        if ((num & (1 << i)) != 0)
        {
            result |= 1 << (log2n - 1 - i);
        }
    }
    return result;
}

// Radix-2 Cooley-Tukey FFT algorithm
void fft(std::vector<std::complex<double>> &inputArray, bool invert = false)
{
    const int n = inputArray.size();
    const int log2n = static_cast<int>(log2(n));
    if (n == 1)
    {
        return;
    }
    if (!isPowerOfTwo(n))
    {
        return;
    }
    // Perform the bit-reversal permutation in-place
    for (int i = 0; i < n; ++i)
    {
        int j = reverseBits(i, log2n);
        if (j > i)
        {
            std::swap(inputArray[i], inputArray[j]);
        }
    }

    // Iterative FFT
    for (int size = 2; size <= n; size *= 2)
    {
        double angle = 2 * M_PI / size * (invert ? -1 : 1);
        std::complex<double> w(1), wn(cos(angle), sin(angle));

        for (int i = 0; i < n; i += size)
        {
            std::complex<double> w_temp(1);
            for (int j = 0; j < size / 2; ++j)
            {
                std::complex<double> u = inputArray[i + j];
                std::complex<double> v = w_temp * inputArray[i + j + size / 2];
                inputArray[i + j] = u + v;
                inputArray[i + j + size / 2] = u - v;
                w_temp *= wn;
            }
            w *= wn;
        }
    }

    // Normalize if inverting
    if (invert)
    {
        for (int i = 0; i < n; ++i)
        {
            inputArray[i] /= n;
        }
    }
}

void swapQuadrants(cv::Mat &magI)
{
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;
    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
    Mat tmp;                            // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void multiplyspectors(cv::Mat &complex1, cv::Mat &complex2)
{
    if (complex1.size() != complex2.size())
    {
        std::cerr << "for multiplyspectors size of image must be equal";
        return;
    }
    for (int i = 0; i < complex1.rows; ++i)
    {
        for (int j = 0; j < complex2.cols; ++j)
        {
            complex1.at<Vec2f>(i, j) = complex1.at<Vec2f>(i, j) * complex2.at<float>(i, j);
        }
    }
}

void lowFilter(cv::Mat &inputImage, size_t radius)
{
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(inputImage.rows);
    int n = cv::getOptimalDFTSize(inputImage.cols);
    cv::copyMakeBorder(inputImage, padded, 0, m - inputImage.rows, 0, n - inputImage.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // Создание комплексного массива для хранения результата преобразования Фурье
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);

    // Применение прямого преобразования Фурье
    cv::dft(complexI, complexI);
    swapQuadrants(complexI);
    Mat highPassFilter = Mat::ones(complexI.rows, complexI.cols, CV_32F);
    int centerX = highPassFilter.cols / 2;
    int centerY = highPassFilter.rows / 2;
    circle(highPassFilter, Point(centerX, centerY), radius, Scalar(0), -1);
    multiplyspectors(complexI, highPassFilter);

    // Расчет магнитуды и логарифмирование
    cv::split(complexI, planes);                    // planes[0] - действительная часть, planes[1] - мнимая часть
    cv::magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
    cv::Mat magI = planes[0];

    // Нормализация для отображения
    magI += cv::Scalar::all(1);
    cv::log(magI, magI);
    
    // Обрезка изображения
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));
    // Нормализация
    cv::normalize(magI, magI, 0, 1, cv::NORM_MINMAX);
    // Сдвинуть нулевую частоту спектра обратно в левый верхний угол
    swapQuadrants(complexI);
    cv::imshow("lowPassFilter", magI);

    cv::Mat reversed;
    cv::idft(complexI, reversed, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    normalize(reversed, reversed, 0, 1, cv::NORM_MINMAX);
    reversed.convertTo(reversed, CV_8U, 255);
    imshow("lowPassReversed", reversed);
}

void highFilter(cv::Mat &inputImage, size_t radius)
{
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(inputImage.rows);
    int n = cv::getOptimalDFTSize(inputImage.cols);
    cv::copyMakeBorder(inputImage, padded, 0, m - inputImage.rows, 0, n - inputImage.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // Создание комплексного массива для хранения результата преобразования Фурье
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);

    // Применение прямого преобразования Фурье
    cv::dft(complexI, complexI);
    swapQuadrants(complexI);
    Mat lowPassFilter = Mat::zeros(complexI.rows, complexI.cols, CV_32F);
    int centerX = lowPassFilter.cols / 2;
    int centerY = lowPassFilter.rows / 2;
    circle(lowPassFilter, Point(centerX, centerY), radius, Scalar(1), -1);
    multiplyspectors(complexI, lowPassFilter);

    // Расчет магнитуды и логарифмирование
    cv::split(complexI, planes);                    // planes[0] - действительная часть, planes[1] - мнимая часть
    cv::magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
    cv::Mat magI = planes[0];

    // Нормализация для отображения
    magI += cv::Scalar::all(1);
    cv::log(magI, magI);
    
    // Обрезка изображения
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));
    // Нормализация
    cv::normalize(magI, magI, 0, 1, cv::NORM_MINMAX);
    // Сдвинуть нулевую частоту спектра обратно в левый верхний угол
    swapQuadrants(complexI);
    cv::imshow("HighFilter", magI);

    cv::Mat reversed;
    cv::idft(complexI, reversed, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    normalize(reversed, reversed, 0, 1, cv::NORM_MINMAX);
    reversed.convertTo(reversed, CV_8U, 255);
    imshow("HighFilterReversed", reversed);
    
}

void displayDFT(cv::Mat &input, const std::string & windowName)
{
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(input.rows);
    int n = cv::getOptimalDFTSize(input.cols);
    cv::copyMakeBorder(input, padded, 0, m - input.rows, 0, n - input.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // Создание комплексного массива для хранения результата преобразования Фурье
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);

    // Применение прямого преобразования Фурье
    cv::dft(complexI, complexI);
    // Расчет магнитуды и логарифмирование
    cv::split(complexI, planes);                    // planes[0] - действительная часть, planes[1] - мнимая часть
    cv::magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
    cv::Mat magI = planes[0];

    // Нормализация для отображения
    magI += cv::Scalar::all(1);
    cv::log(magI, magI);

    // Обрезка изображения
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));
    swapQuadrants(magI);
    // Нормализация
    cv::normalize(magI, magI, 0, 1, cv::NORM_MINMAX);
    cv::imshow(windowName, magI);

    cv::Mat reversed;
    cv::idft(complexI, reversed, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    normalize(reversed, reversed, 0, 1, cv::NORM_MINMAX);
    reversed.convertTo(reversed, CV_8U, 255);
    imshow(windowName + " revers", reversed);

}

std::vector<std::complex<double>> convertMatToVector(cv::Mat image)
{
    std::vector<uchar> imageVector(image.begin<uchar>(), image.end<uchar>());
    std::vector<std::complex<double>> complexVector;

    for (const auto &val : imageVector)
    {
        complexVector.push_back(std::complex<double>(val, 0));
    }
    return complexVector;
}

void expandCanvas(cv::Mat& img32FC1, cv::Size size)
{
	if (img32FC1.empty())
	{
		return;
	}
	if (img32FC1.rows > size.height || img32FC1.cols > size.width)
	{
		return;
	}

	cv::Mat expandedImg(size, CV_32FC1, cv::Scalar(0));
	cv::Mat tempROI(expandedImg, cv::Rect(0, 0, img32FC1.cols, img32FC1.rows));
	img32FC1.copyTo(tempROI);
	img32FC1 = expandedImg.clone();

	return;
}

void correlation(cv::Mat& inputImage, cv::Mat& sample, const double& threshold_minus)
{
	if (inputImage.empty() | sample.empty())
	{
		return;
	}
    inputImage.convertTo(inputImage, CV_32FC1);
    sample.convertTo(sample, CV_32FC1);
	cv::Scalar avg_value_1;
	cv::Mat deviation_1;
	cv::meanStdDev(sample, avg_value_1, deviation_1);  // Вычисление среднего значения и стандратного отклонения пикселей в sample

	cv::Scalar avg_value_2;
	cv::Mat deviation_2;
	cv::meanStdDev(inputImage, avg_value_2, deviation_2); // Вычисление среднего значения и стандратного отклонения пикселей в изображении

    // Центирование данных относительно их среднего значения
	sample -= avg_value_1; 
	inputImage -= avg_value_2;

    Size dftSize;
    dftSize.width = getOptimalDFTSize(inputImage.cols + sample.cols - 1);
    dftSize.height = getOptimalDFTSize(inputImage.rows + sample.rows - 1);

	expandCanvas(sample, dftSize);
    expandCanvas(inputImage, dftSize);

    Mat dftOfImage(inputImage.size(), CV_32FC2);
    dft(inputImage, dftOfImage, DFT_COMPLEX_OUTPUT);
    Mat dftOfSample(inputImage.size(), CV_32FC2);
    dft(sample, dftOfSample, DFT_COMPLEX_OUTPUT);

    Mat dftCorrelation(inputImage.size(), CV_32FC2);
	cv::mulSpectrums(dftOfImage, dftOfSample, dftCorrelation, 0, 1);

	idft(dftCorrelation, dftCorrelation, DFT_INVERSE | DFT_REAL_OUTPUT);
	normalize(dftCorrelation, dftCorrelation, 0, 1, cv::NORM_MINMAX);

	cv::Mat withoutThreshold;
	dftCorrelation.convertTo(withoutThreshold, CV_8UC1, 255);
	cv::imshow("beforeThreshold", withoutThreshold);

	double thresh;
	cv::minMaxLoc(dftCorrelation, NULL, &thresh);
	thresh -= threshold_minus;
	cv::threshold(dftCorrelation, dftCorrelation, thresh, 1, cv::THRESH_BINARY);
    cv::Mat colorImage(dftCorrelation.size(), CV_8UC3);
    cv::cvtColor(dftCorrelation, colorImage, cv::COLOR_GRAY2BGR);
    std::vector<cv::Point> nonzeroPixels;
    cv::findNonZero(dftCorrelation, nonzeroPixels);
    int radius = 50; 
    cv::Scalar color(0, 0, 255); 

    for (const auto& point : nonzeroPixels) {
        cv::circle(colorImage, point, radius, color, 1);
    }
    dftCorrelation.convertTo(dftCorrelation, CV_8UC1, 255);
    imshow("foundedSample", dftCorrelation);
    imshow("Result Image", colorImage);
}

void test_dft()
{
    cv::Mat image = cv::imread("D:/repositories/OpenCV/images/photo_2023-12-02_19-40-35.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        std::cout << "Could't open image" << std::endl;
    }
    cv::resize(image, image, cv::Size(128, 128));
    imshow("Input Image", image);
    image.convertTo(image, CV_32F);
    Mat padded; // expand input image to optimal size
    int m = getOptimalDFTSize(image.rows);
    int n = getOptimalDFTSize(image.cols); // on the border add zero values
    copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);
    complexI = DFT_IMAGE(image);
    split(complexI, planes);                    // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
    Mat magI = planes[0];
    magI += Scalar::all(1); // switch to logarithmic scale
    log(magI, magI);
    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    swapQuadrants(magI);
    normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
                                              // viewable image form (float between values 0 and 1).
                                              // Show the result
    imshow("spectrum magnitude", magI);
    cv::Mat reversed;
    reversed = IDFT_IMAGE(complexI);
    normalize(reversed, reversed, 0, 1, cv::NORM_MINMAX);
    reversed.convertTo(reversed, CV_8U, 255);
    imshow("revers", reversed);
}

void test_fft()
{
    cv::Mat image = cv::imread("D:/repositories/OpenCV/images/photo_2023-12-02_19-40-35.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        std::cout << "Could't open image" << std::endl;
    }
    cv::resize(image, image, cv::Size(128, 128));
    imshow("input Image", image);
    std::vector<std::complex<double>> inputArray = convertMatToVector(image);
    Mat padded; // expand input image to optimal size
    int m = getOptimalDFTSize(image.rows);
    int n = getOptimalDFTSize(image.cols); // on the border add zero values
    copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);
    fft(inputArray, false);
    complexI = convertComplexVectorToMat(inputArray, m, n);
    split(complexI, planes);                    // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
    Mat magI = planes[0];
    magI += Scalar::all(1); // switch to logarithmic scale
    log(magI, magI);
    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    swapQuadrants(magI);
    normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
                                              // viewable image form (float between values 0 and 1).
                                              // Show the result
    imshow("spectrum magnitude", magI);
    fft(inputArray, true);
    cv::Mat reversed;
    Mat planesOutput[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    merge(planesOutput, 2, reversed);
    reversed = convertComplexVectorToMat(inputArray, m, n);
    split(reversed, planesOutput);
    Mat output = planesOutput[0];
    cv::normalize(output, output, 0, 1, cv::NORM_MINMAX); // Нормализуем значения для отображения
    cv::imshow("Reversed Image", output);
}

void test_time()
{
    cv::Mat image = cv::imread("D:/repositories/OpenCV/images/photo_2023-12-02_19-40-35.jpg", cv::IMREAD_GRAYSCALE);
    std::vector<std::complex<double>> InputArray = convertMatToVector(image);
    
    if (image.empty())
    {
        std::cout << "Could't open image" << std::endl;
    }
    cv::resize(image, image, cv::Size(128, 128));
    imshow("Input Image", image);
    auto start_custom = std::chrono::high_resolution_clock::now();
    DFT_IMAGE(image);
    auto end_custom = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_custom = end_custom - start_custom;
    std::cout << "Time wrapped by DFT " << elapsed_custom.count() << " seconds" << std::endl;

    auto start_custom_radix = std::chrono::high_resolution_clock::now();
    fft(InputArray, false);
    auto end_custom_radix = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_custom_radix = end_custom_radix - start_custom_radix;
    std::cout << "Time wrapped by FFT " << elapsed_custom_radix.count() << " seconds" << std::endl;
    auto start_custom_cv = std::chrono::high_resolution_clock::now();

    auto end_custom_cv = std::chrono::high_resolution_clock::now();
    image.convertTo(image, CV_32F);
    cv::dft(image, image, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    std::chrono::duration<double> elapsed_custom_cv = end_custom_cv - start_custom_cv;
    std::cout << "Time wrapped by FFT " << elapsed_custom_cv.count() << " seconds" << std::endl;
}

void test_filters()
{
    cv::Mat image = cv::imread("D:/repositories/OpenCV/images/220px-Lenna.png", cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        std::cout << "Could't open image" << std::endl;
    }
    // Создание ядер для сверток
    cv::Mat sobelX, sobelY;
    cv::Sobel(image, sobelX, CV_32F, 1, 0);
    cv::Sobel(image, sobelY, CV_32F, 0, 1);

    cv::Mat boxFilter;
    cv::boxFilter(image, boxFilter, -1, cv::Size(3, 3));

    cv::Mat laplacian;
    cv::Laplacian(image, laplacian, CV_32F);

    // Отображаем магнитуду Фурье для каждого изображения
    displayDFT(image, "Original Image DFT Magnitude");
    displayDFT(sobelX, "Sobel X DFT Magnitude");
    displayDFT(sobelY, "Sobel Y DFT Magnitude");
    displayDFT(boxFilter, "Box Filter DFT Magnitude");
    displayDFT(laplacian, "Laplacian DFT Magnitude");

    // Отображение исходного изображения и сверток
    cv::imshow("Original Image", image);
    cv::imshow("Sobel X", sobelX);
    cv::imshow("Sobel Y", sobelY);
    cv::imshow("Box Filter", boxFilter);
    cv::imshow("Laplacian", laplacian);
}

void test_correlate()
{
    cv::Mat image = cv::imread("D:/repositories/OpenCV/images/cat.jpg", cv::IMREAD_GRAYSCALE);
    imshow("original", image);
    Mat sample = imread("D:/repositories/OpenCV/images/glaz.jpg", IMREAD_GRAYSCALE);
    imshow("Sample", sample);
    correlation(image, sample, 0.02);
}

void test_high_low_pass()
{
    cv::Mat image = cv::imread("D:/repositories/OpenCV/images/220px-Lenna.png", cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        std::cout << "Could't open image" << std::endl;
    }
    imshow("original", image);
    // Apply lowFilter
    lowFilter(image, 30);
    // Apply Highfilter
    highFilter(image, 30);
    
    // displayDFT(image, "Original Image DFT Magnitude", false, true, 60);
}

int main()
{
    // test_dft();
    // test_fft();
    // test_filters();
    // test_high_low_pass();
    test_correlate();
    waitKey(0);
    return 0;
}