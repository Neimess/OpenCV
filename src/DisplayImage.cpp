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

cv::Mat convertComplexVectorToMat(const std::vector<std::complex<double>> &complexVector, size_t _rows)
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

// Wrapper function for the FFT
std::vector<std::complex<double>> fft(std::vector<std::complex<double>> &inputArray)
{
    fft(inputArray, false);
    return inputArray;
}

// Wrapper function for the inverse FFT
std::vector<std::complex<double>> ifft(std::vector<std::complex<double>> &inputArray)
{
    fft(inputArray, true);
    return inputArray;
}

void krasivSpectr(cv::Mat &magI)
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

void displayDFT(cv::Mat& input, const std::string& windowName) {
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
    cv::split(complexI, planes);          // planes[0] - действительная часть, planes[1] - мнимая часть
    cv::magnitude(planes[0], planes[1], planes[0]);  // planes[0] = magnitude
    cv::Mat magI = planes[0];

    // Нормализация для отображения
    magI += cv::Scalar::all(1);
    cv::log(magI, magI);

    // Обрезка изображения
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

    krasivSpectr(magI);
    // Нормализация
    cv::normalize(magI, magI, 0, 1, cv::NORM_MINMAX);
    cv::imshow(windowName, magI);
}
int main()
{
    cv::Mat image = cv::imread("D:/repositories/OpenCV/photo_2023-12-02_19-40-35.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()){
        std::cout << "Could't open image" << std::endl;
    }
    cv::resize(image, image, cv::Size(128, 128));
    std::vector<uchar> imageVector(image.begin<uchar>(), image.end<uchar>());
    std::vector<std::complex<double>> complexVector;

    for (const auto &val : imageVector)
    {
        complexVector.push_back(std::complex<double>(val, 0));
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

    

    cv::waitKey(0);


    // Mat padded; // expand input image to optimal size
    // int m = getOptimalDFTSize(image.rows);
    // int n = getOptimalDFTSize(image.cols); // on the border add zero values
    // copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));
    // Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    // Mat complexI;
    // merge(planes, 2, complexI); // Add to the expanded another plane with zeros
    // dft(complexI, complexI);            // this way the result may fit in the source matrix
    // // std::vector<std::complex<double>> complexVectorDFT = DFT(complexVector);
    // // complexI = convertComplexVectorToMat(complexVectorDFT, image.rows);
    // fft(complexVector);
    // complexI = convertComplexVectorToMat(complexVector, image.rows);
    // // compute the magnitude and switch to logarithmic scale
    // // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    // split(complexI, planes);                    // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    // magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
    // Mat magI = planes[0];
    // magI += Scalar::all(1); // switch to logarithmic scale
    // log(magI, magI);
    // // crop the spectrum, if it has an odd number of rows or columns
    // magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    // // rearrange the quadrants of Fourier image  so that the origin is at the image center
    // krasivSpectr(magI);
    // normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
    //                                           // viewable image form (float between values 0 and 1).
    // imshow("Input Image", image);             // Show the result
    // imshow("spectrum magnitude", magI);

    // Mat reversed;
    // // std::vector<std::complex<double>> complexVectorIDFT = IDFT(complexVectorDFT);
    // // reversed = convertComplexVectorToMat(complexVectorIDFT, image.rows);
    // cv::idft(complexI, reversed, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    // normalize(reversed, reversed, 0, 1, cv::NORM_MINMAX);
    // reversed.convertTo(reversed, CV_8U, 255);
    // imshow("reversed", reversed);
    // waitKey(0);
    // auto start_custom = std::chrono::high_resolution_clock::now();
    // std::vector<std::complex<double>> result = DFT(complexVector);
    // auto end_custom = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed_custom = end_custom - start_custom;
    // std::cout << "Time wrapped by DFT " << elapsed_custom.count() << " seconds" << std::endl;
    
    // std::cout << std::endl;
    // auto start_radix = std::chrono::high_resolution_clock::now();
    // std::vector<std::complex<double>> radix = fft(complexVector);
    // auto end_radix = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed_radix = end_radix - start_radix;
    // std::cout << "Time wrapped by Radix Method FFT " << elapsed_radix.count() << "seconds" << std::endl;

    // auto start_cv_fft = std::chrono::high_resolution_clock::now();
    // copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));
    // image.convertTo(image, CV_32F);
    // cv::dft(image, image, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);            // this way the result may fit in the source matrix
    // auto end_cv_fft = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed_cv = end_cv_fft - start_cv_fft;
    // std::cout << "Time wrapped by Radix Method CV FFT " << elapsed_cv.count() << " seconds" << std::endl;
    return 0;
}