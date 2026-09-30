#include <sycl/sycl.hpp>
#include <mpi.h>

#include <iostream>
#include <vector>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 1000000;

    sycl::queue q(sycl::cpu_selector_v);

    std::vector<int> data(N, 1);

    int *device_data =
        sycl::malloc_device<int>(N, q);

    q.memcpy(
        device_data,
        data.data(),
        N * sizeof(int)
    ).wait();

    int local_sum = 0;

    sycl::buffer<int> result_buf(
        &local_sum,
        sycl::range<1>(1)
    );

    q.submit([&](sycl::handler &h) {

        auto result =
            result_buf.get_access<
                sycl::access::mode::write>(h);

        h.parallel_for(
            sycl::range<1>(1),
            [=](sycl::id<1>) {

                int sum = 0;

                for (int i = 0; i < N; ++i)
                    sum += device_data[i];

                result[0] = sum;
            }
        );

    }).wait();

    int global_sum = 0;

    MPI_Reduce(
        &local_sum,
        &global_sum,
        1,
        MPI_INT,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {

        std::cout
            << "MPI ranks : "
            << size
            << std::endl;

        std::cout
            << "Global sum : "
            << global_sum
            << std::endl;
    }

    sycl::free(device_data, q);

    MPI_Finalize();

    return 0;
}
