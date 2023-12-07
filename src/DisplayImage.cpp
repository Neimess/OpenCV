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

                    dftReal.at<float>(k, l) += inputImage.at<float>(m, n) * std::cos(angle) - inputImage.at<float>(m, n) * std::sin(angle);
                    dftImag.at<float>(k, l) += inputImage.at<float>(m, n) * std::sin(angle) + inputImage.at<float>(m, n) * std::cos(angle);
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

cv::Mat convertComplexVectorToMat(const std::vector<std::complex<float>> &complexVector, size_t _rows)
{
    int rows = 1; // одна строка, так как у нас вектор
    int cols = complexVector.size();
    cv::Mat resultMat(1, cols, CV_64FC2); // создаем матрицу для хранения комплексных чисел

    for (int i = 0; i < cols; ++i)
    {
        resultMat.at<cv::Vec2d>(0, i)[0] = complexVector[i].real(); // реальная часть комплексного числа
        resultMat.at<cv::Vec2d>(0, i)[1] = complexVector[i].imag(); // мнимая часть комплексного числа
    }
    cv::Mat reshapedMat = resultMat.reshape(0, _rows);

    return reshapedMat;
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

// Recursive radix-2 FFT implementation
void fft(std::vector<std::complex<double>> &inputArray, bool invert)
{
    int n = inputArray.size();
    if (n <= 1)
    {
        return;
    }
    if (!isPowerOfTwo(n))
    {
        return;
    }

    std::vector<std::complex<double>> a0(n / 2), a1(n / 2);
    for (int i = 0, j = 0; i < n; i += 2, ++j)
    {
        a0[j] = inputArray[i];
        a1[j] = inputArray[i + 1];
    }

    fft(a0, invert);
    fft(a1, invert);

    double angle = 2 * M_PI / n * (invert ? -1 : 1);
    std::complex<double> w(1), wn(cos(angle), sin(angle));

    for (int i = 0; i < n / 2; ++i)
    {
        std::complex<double> t = w * a1[i];
        inputArray[i] = a0[i] + t;
        inputArray[i + n / 2] = a0[i] - t;
        if (invert)
        {
            inputArray[i] /= 2;
            inputArray[i + n / 2] /= 2;
        }
        w *= wn;
    }
}

void fft1D(std::vector<std::complex<double>> &inputArray, bool invert)
{
    int n = inputArray.size();
    if (n <= 1 || !isPowerOfTwo(n))
    {
        return;
    }

    std::vector<std::complex<double>> a0(n / 2), a1(n / 2);
    for (int i = 0, j = 0; i < n; i += 2, ++j)
    {
        a0[j] = inputArray[i];
        a1[j] = inputArray[i + 1];
    }

    fft1D(a0, invert);
    fft1D(a1, invert);

    double angle = 2 * M_PI / n * (invert ? -1 : 1);
    std::complex<double> w(1), wn(cos(angle), sin(angle));

    for (int i = 0; i < n / 2; ++i)
    {
        std::complex<double> t = w * a1[i];
        inputArray[i] = a0[i] + t;
        inputArray[i + n / 2] = a0[i] - t;
        if (invert)
        {
            inputArray[i] /= 2;
            inputArray[i + n / 2] /= 2;
        }
        w *= wn;
    }
}

void fft2D(cv::Mat &inputMatrix, bool invert)
{
    int rows = inputMatrix.rows;
    int cols = inputMatrix.cols;

    if (!isPowerOfTwo(rows) || !isPowerOfTwo(cols))
    {
        return;
    }

    // Применяем FFT к каждой строке
    for (int i = 0; i < rows; ++i)
    {
        std::vector<std::complex<double>> row(cols);
        for (int j = 0; j < cols; ++j)
        {
            row[j] = inputMatrix.at<std::complex<double>>(i, j);
        }
        fft1D(row, invert);
        for (int j = 0; j < cols; ++j)
        {
            inputMatrix.at<std::complex<double>>(i, j) = row[j];
        }
    }

    // Применяем FFT к каждому столбцу
    for (int j = 0; j < cols; ++j)
    {
        std::vector<std::complex<double>> column(rows);
        for (int i = 0; i < rows; ++i)
        {
            column[i] = inputMatrix.at<std::complex<double>>(i, j);
        }
        fft1D(column, invert);
        for (int i = 0; i < rows; ++i)
        {
            inputMatrix.at<std::complex<double>>(i, j) = column[i];
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

void highPassFilter(cv::Mat &complexI, size_t radius)
{
    Mat highPassFilter = Mat::ones(complexI.rows, complexI.cols, CV_32F);
    int centerX = highPassFilter.cols / 2;
    int centerY = highPassFilter.rows / 2;
    circle(highPassFilter, Point(centerX, centerY), radius, Scalar(0), -1);
    multiplyspectors(complexI, highPassFilter);
}

void lowPassFilter(cv::Mat &complexI, size_t radius)
{
    Mat lowPassFilter = Mat::zeros(complexI.rows, complexI.cols, CV_32F);
    int centerX = lowPassFilter.cols / 2;
    int centerY = lowPassFilter.rows / 2;
    circle(lowPassFilter, Point(centerX, centerY), radius, Scalar(1), -1);
    multiplyspectors(complexI, lowPassFilter);
}

void displayDFT(cv::Mat &input, const std::string & windowName,
                        bool applyLowPassFilter = false,
                        bool applyHighPassFilter = false,
                        int borderRadius = 60)
{
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(input.rows);
    int n = cv::getOptimalDFTSize(input.cols);
    cv::copyMakeBorder(input, padded, 0, m - input.rows, 0, n - input.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // Создание комплексного массива для хранения результата преобразования Фурье
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_64F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);

    // Применение прямого преобразования Фурье
    cv::dft(complexI, complexI);
    if (applyHighPassFilter) highPassFilter(complexI, borderRadius);
    if (applyLowPassFilter) lowPassFilter(complexI, borderRadius);
    // Расчет магнитуды и логарифмирование
    cv::split(complexI, planes);                    // planes[0] - действительная часть, planes[1] - мнимая часть
    cv::magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
    cv::Mat magI = planes[0];

    // Нормализация для отображения
    magI += cv::Scalar::all(1);
    cv::log(magI, magI);

    // Обрезка изображения
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));
    if (applyHighPassFilter || applyLowPassFilter) swapQuadrants(magI);
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

int correlate(Mat &input, Mat &sample, const float thresholdMul)
{
    if (sample.empty())
    {
        return -1;
    }
    if (sample.empty())
    {
        return -2;
    }
    sample.convertTo(sample, CV_32FC1);
    Size dftSize;
    dftSize.width = getOptimalDFTSize(input.cols + sample.cols - 1);
    dftSize.height = getOptimalDFTSize(input.rows + sample.rows - 1);
    Mat expandedImage(dftSize, CV_32FC1, Scalar(0));
    Mat tempROI(expandedImage, Rect(0, 0, input.cols, input.rows));
    input.copyTo(tempROI);
    Mat expandedSample(dftSize, CV_32FC1, Scalar(0));
    Mat tempROI2(expandedSample, Rect(0, 0, sample.cols, sample.rows));
    sample.copyTo(tempROI2);

    Mat dftOfImage(expandedImage.cols, expandedImage.rows, CV_32FC2);
    dft(expandedImage, dftOfImage, DFT_COMPLEX_OUTPUT);
    Mat dftOfSample(expandedSample.cols, expandedSample.rows, CV_32FC2);
    dft(expandedSample, dftOfSample, DFT_COMPLEX_OUTPUT);

    Mat dftCorrelation(dftSize, CV_32FC2);
    mulSpectrums(dftOfImage, dftOfSample, dftCorrelation, true);

    Mat uncroppedOutputImage(dftSize, CV_32FC2);
    idft(dftCorrelation, uncroppedOutputImage, DFT_INVERSE | DFT_REAL_OUTPUT);
    normalize(uncroppedOutputImage, uncroppedOutputImage, 0.0f, 1.0f, NORM_MINMAX);

    Mat croppedImage(uncroppedOutputImage, Rect(0, 0, input.cols, input.rows));
    imshow("Before threshold", croppedImage);

    float maxIntensity = 0;
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            const float currentPixel = croppedImage.at<float>(i, j);
            if (currentPixel > maxIntensity)
                maxIntensity = currentPixel;
        }
    }
    const float threshold = thresholdMul * maxIntensity;
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            const float currentPixel = croppedImage.at<float>(i, j);
            if (currentPixel < threshold)
                croppedImage.at<float>(i, j) = 0;
        }
    }
    imshow("cropped image", croppedImage);
    waitKey(0);
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
    complexI = DFT_IMAGE(complexI);
    // fft2D(image, false);
    std::cout << image;
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
    waitKey(0);
}

void test_fft()
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
    complexI = DFT_IMAGE(complexI);
    fft2D(image, false);
    std::cout << image;
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
    waitKey(0);
}

void test_time() {
    cv::Mat image = cv::imread("D:/repositories/OpenCV/images/photo_2023-12-02_19-40-35.jpg", cv::IMREAD_GRAYSCALE);
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
    fft2D(image, false);
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

void test_filters() {
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
    cv::boxFilter(image, boxFilter, -1, cv::Size(3,3));

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

void test_correlate() {
    cv::Mat image = cv::imread("D:/repositories/OpenCV/images/nomera.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        std::cout << "Could't open image" << std::endl;
    }
    image.convertTo(image, CV_32F);
    cv::resize(image, image, cv::Size(3,3));
    imshow("original", image);
    image.convertTo(image, CV_32FC1);
    Mat sample = imread("D:/repositories/OpenCV/images/symbol_6.jpg", IMREAD_GRAYSCALE);

    imshow("Sample", sample);
    correlate(image, sample, 0.999);
}

void test_filters() {
    cv::Mat image = cv::imread("D:/repositories/OpenCV/images/220px-Lenna.png", cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        std::cout << "Could't open image" << std::endl;
    }
    imshow("original", image);
    // Apply lowPassFilter
    displayDFT(image, "Original Image DFT Magnitude", true, false, 60);
    // Apply HighPassFilter
    // displayDFT(image, "Original Image DFT Magnitude", false, true, 60);
   
}
int main()
{
    test_fft();
    waitKey(0);
    return 0;
}