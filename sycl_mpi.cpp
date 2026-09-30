#include <sycl/sycl.hpp>
#include <mpi.h>

#include <iostream>

int main(int argc, char *argv[])
{
    // --------------------------------------------------------
    // Initialize MPI
    // --------------------------------------------------------
    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --------------------------------------------------------
    // Create SYCL queue
    // --------------------------------------------------------
    sycl::queue q(
        sycl::default_selector_v,
        [](sycl::exception_list exceptions) {
            for (auto &e : exceptions) {
                try {
                    std::rethrow_exception(e);
                }
                catch (const sycl::exception &ex) {
                    std::cerr
                        << "SYCL asynchronous exception: "
                        << ex.what()
                        << std::endl;
                }
            }
        }
    );

    // --------------------------------------------------------
    // Print rank/device information
    // --------------------------------------------------------
    auto device = q.get_device();

    std::cout
        << "Rank " << rank
        << " / " << size
        << " running on "
        << device.get_info<sycl::info::device::name>()
        << std::endl;

    // --------------------------------------------------------
    // Each rank calculates:
    //
    // rank + rank + rank + ... 10 times
    //
    // Example:
    // rank 0 -> 0
    // rank 1 -> 10
    // rank 2 -> 20
    // rank 3 -> 30
    // --------------------------------------------------------

    const int N = 10;

    int *data = sycl::malloc_shared<int>(N, q);

    for (int i = 0; i < N; i++) {
        data[i] = rank;
    }

    // --------------------------------------------------------
    // SYCL calculation
    // --------------------------------------------------------

    q.parallel_for(
        sycl::range<1>(N),
        [=](sycl::id<1> i) {
            data[i] = data[i] + rank;
        }
    ).wait();

    // --------------------------------------------------------
    // Calculate local sum
    // --------------------------------------------------------

    int local_sum = 0;

    for (int i = 0; i < N; i++) {
        local_sum += data[i];
    }

    std::cout
        << "Rank " << rank
        << " local sum = "
        << local_sum
        << std::endl;

    // --------------------------------------------------------
    // MPI reduction
    //
    // Add local sums from all MPI ranks
    // --------------------------------------------------------

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

    // --------------------------------------------------------
    // Rank 0 prints final result
    // --------------------------------------------------------

    if (rank == 0) {

        std::cout
            << "\n====================================\n";

        std::cout
            << "MPI + SYCL RESULT\n";

        std::cout
            << "====================================\n";

        std::cout
            << "Number of MPI ranks : "
            << size
            << std::endl;

        std::cout
            << "Global sum          : "
            << global_sum
            << std::endl;

        std::cout
            << "====================================\n";
    }

    // --------------------------------------------------------
    // Free SYCL memory
    // --------------------------------------------------------

    sycl::free(data, q);

    // --------------------------------------------------------
    // Finalize MPI
    // --------------------------------------------------------

    MPI_Finalize();

    return 0;
}
