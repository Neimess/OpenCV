#include <complex>
#include <iostream>
#include <vector>
#include <math.h>

unsigned int reverseBits(unsigned int num, int log2n) {
    unsigned int result = 0;
    for (int i = 0; i < log2n; ++i) {
        if ((num & (1 << i)) != 0) {
            result |= 1 << (log2n - 1 - i);
        }
    }
    return result;
}

// Radix-2 Cooley-Tukey FFT algorithm
void fft(std::vector<std::complex<double>>& inputArray, bool invert = false) {
    const int n = inputArray.size();
    const int log2n = static_cast<int>(log2(n));

    // Perform the bit-reversal permutation in-place
    for (int i = 0; i < n; ++i) {
        int j = reverseBits(i, log2n);
        if (j > i) {
            std::swap(inputArray[i], inputArray[j]);
        }
    }

    // Iterative FFT
    for (int size = 2; size <= n; size *= 2) {
        double angle = 2 * M_PI / size * (invert ? -1 : 1);
        std::complex<double> w(1), wn(cos(angle), sin(angle));

        for (int i = 0; i < n; i += size) {
            std::complex<double> w_temp(1);
            for (int j = 0; j < size / 2; ++j) {
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
    if (invert) {
        for (int i = 0; i < n; ++i) {
            inputArray[i] /= n;
        }
    }
}

int main() {
    // Example usage
    std::vector<std::complex<double>> inputArray = {1, 2, 3, 4};
    
    std::cout << "Original array:" << std::endl;
    for (const auto& element : inputArray) {
        std::cout << element << " ";
    }
    std::cout << std::endl;

    // Apply FFT
    fft(inputArray);

    std::cout << "FFT result:" << std::endl;
    for (const auto& element : inputArray) {
        std::cout << element << " ";
    }
    std::cout << std::endl;

    // Apply inverse FFT
    fft(inputArray, true);

    std::cout << "Inverse FFT result:" << std::endl;
    for (const auto& element : inputArray) {
        std::cout << element << " ";
    }
    std::cout << std::endl;

    return 0;
}