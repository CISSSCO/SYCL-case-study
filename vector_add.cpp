#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

int main()
{
    const size_t N = 1000000;

    std::vector<float> A(N, 1.0f);
    std::vector<float> B(N, 2.0f);
    std::vector<float> C(N, 0.0f);

    sycl::queue q;

    float *d_A = sycl::malloc_device<float>(N, q);
    float *d_B = sycl::malloc_device<float>(N, q);
    float *d_C = sycl::malloc_device<float>(N, q);

    q.memcpy(d_A, A.data(), N * sizeof(float));
    q.memcpy(d_B, B.data(), N * sizeof(float));

    q.wait();

    q.parallel_for(
        sycl::range<1>(N),
        [=](sycl::id<1> i) {
            d_C[i] = d_A[i] + d_B[i];
        }
    ).wait();

    q.memcpy(C.data(), d_C, N * sizeof(float)).wait();

    std::cout << "C[0] = "
              << C[0]
              << std::endl;

    sycl::free(d_A, q);
    sycl::free(d_B, q);
    sycl::free(d_C, q);

    return 0;
}
