#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

int main() {

    constexpr size_t N = 30000;

    std::cout << "========================================\n";
    std::cout << "      SYCL GPU MATRIX MULTIPLICATION\n";
    std::cout << "========================================\n";

    try {

        sycl::queue q(
            sycl::gpu_selector_v,
            [](sycl::exception_list exceptions) {
                for (auto &e : exceptions) {
                    try {
                        std::rethrow_exception(e);
                    }
                    catch (const sycl::exception &ex) {
                        std::cerr << "Asynchronous SYCL exception: "
                                  << ex.what() << '\n';
                    }
                }
            }
        );

        auto device = q.get_device();

        std::cout << "\nDevice Information\n";
        std::cout << "------------------\n";
        std::cout << "Device : "
                  << device.get_info<sycl::info::device::name>()
                  << '\n';

        std::cout << "Vendor : "
                  << device.get_info<sycl::info::device::vendor>()
                  << '\n';

        std::cout << "Driver : "
                  << device.get_info<sycl::info::device::driver_version>()
                  << '\n';

        std::cout << "Matrix size : "
                  << N << " x " << N << '\n';

        const size_t total = N * N;

        std::vector<float> A(total);
        std::vector<float> B(total);
        std::vector<float> C(total, 0.0f);

        // Initialize matrices
        for (size_t i = 0; i < total; ++i) {
            A[i] = 1.0f;
            B[i] = 2.0f;
        }

        // Allocate memory on GPU
        float *d_A = sycl::malloc_device<float>(total, q);
        float *d_B = sycl::malloc_device<float>(total, q);
        float *d_C = sycl::malloc_device<float>(total, q);

        if (!d_A || !d_B || !d_C) {
            std::cerr << "GPU memory allocation failed.\n";
            return 1;
        }

        // Copy input matrices to GPU
        q.memcpy(d_A, A.data(), total * sizeof(float));
        q.memcpy(d_B, B.data(), total * sizeof(float));
        q.wait();

        std::cout << "\nRunning matrix multiplication...\n";

        auto start = std::chrono::high_resolution_clock::now();

        /*
         * C = A x B
         *
         * Each SYCL work-item calculates one element of C.
         */
        q.parallel_for(
            sycl::range<2>(N, N),
            [=](sycl::id<2> idx) {

                size_t row = idx[0];
                size_t col = idx[1];

                float sum = 0.0f;

                for (size_t k = 0; k < N; ++k) {
                    sum += d_A[row * N + k] *
                           d_B[k * N + col];
                }

                d_C[row * N + col] = sum;
            }
        );

        q.wait();

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;

        // Copy result back to CPU
        q.memcpy(C.data(), d_C, total * sizeof(float)).wait();

        // Verify result
        bool correct = true;

        for (size_t i = 0; i < total; ++i) {
            if (C[i] != 2.0f * static_cast<float>(N)) {
                correct = false;
                break;
            }
        }

        double operations =
            2.0 * static_cast<double>(N) *
            static_cast<double>(N) *
            static_cast<double>(N);

        double gflops =
            operations / elapsed.count() / 1.0e9;

        std::cout << "\n========================================\n";
        std::cout << "             RESULTS\n";
        std::cout << "========================================\n";

        std::cout << "C[0][0]       : " << C[0] << '\n';
        std::cout << "C[N-1][N-1]   : " << C[total - 1] << '\n';

        std::cout << "Verification   : "
                  << (correct ? "PASSED" : "FAILED")
                  << '\n';

        std::cout << "Execution time : "
                  << elapsed.count()
                  << " seconds\n";

        std::cout << "Performance    : "
                  << gflops
                  << " GFLOPS\n";

        std::cout << "========================================\n";

        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_C, q);

    }
    catch (const sycl::exception &e) {

        std::cerr << "\nSYCL exception:\n"
                  << e.what() << '\n';

        return 1;
    }

    return 0;
}
