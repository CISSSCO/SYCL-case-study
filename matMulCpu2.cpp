#include <sycl/sycl.hpp>
#include <mpi.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

constexpr int TILE = 16;

// ------------------------------------------------------------
// MPI + SYCL Tiled Matrix Multiplication
// ------------------------------------------------------------

int main(int argc, char **argv) {

    // --------------------------------------------------------
    // MPI initialization
    // --------------------------------------------------------

    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --------------------------------------------------------
    // Matrix size
    // --------------------------------------------------------

    constexpr size_t N = 4096;

    if (N % TILE != 0) {

        if (rank == 0) {
            std::cerr
                << "ERROR: N must be divisible by TILE.\n";
        }

        MPI_Finalize();
        return 1;
    }

    // --------------------------------------------------------
    // Create SYCL queue
    // --------------------------------------------------------

    sycl::queue q;

    try {

        q = sycl::queue(
            sycl::default_selector_v,
            [](sycl::exception_list exceptions) {

                for (const auto &e : exceptions) {

                    try {
                        std::rethrow_exception(e);
                    }
                    catch (const sycl::exception &ex) {

                        std::cerr
                            << "Asynchronous SYCL exception: "
                            << ex.what()
                            << '\n';
                    }
                }
            }
        );
    }
    catch (const sycl::exception &e) {

        if (rank == 0) {
            std::cerr
                << "Failed to create SYCL queue:\n"
                << e.what()
                << '\n';
        }

        MPI_Finalize();
        return 1;
    }

    // --------------------------------------------------------
    // Device information
    // --------------------------------------------------------

    auto device = q.get_device();

    if (rank == 0) {

        std::cout
            << "========================================\n"
            << "     MPI + SYCL MATRIX MULTIPLICATION\n"
            << "========================================\n\n";

        std::cout
            << "MPI processes : "
            << size
            << '\n';

        std::cout
            << "Matrix size   : "
            << N
            << " x "
            << N
            << '\n';

        std::cout
            << "Tile size     : "
            << TILE
            << " x "
            << TILE
            << "\n\n";
    }

    std::cout
        << "[Rank "
        << rank
        << "] Host   : ";

    char hostname[MPI_MAX_PROCESSOR_NAME];
    int hostname_len = 0;

    MPI_Get_processor_name(
        hostname,
        &hostname_len
    );

    std::cout
        << hostname
        << '\n';

    std::cout
        << "[Rank "
        << rank
        << "] Device : "
        << device.get_info<
            sycl::info::device::name>()
        << '\n';

    std::cout
        << "[Rank "
        << rank
        << "] Vendor : "
        << device.get_info<
            sycl::info::device::vendor>()
        << "\n\n";

    // --------------------------------------------------------
    // Divide rows between MPI ranks
    // --------------------------------------------------------

    const size_t rows_per_rank = N / size;

    if (N % size != 0) {

        if (rank == 0) {
            std::cerr
                << "ERROR: N must be divisible by "
                << "number of MPI ranks.\n";
        }

        MPI_Finalize();
        return 1;
    }

    const size_t local_rows = rows_per_rank;

    // --------------------------------------------------------
    // Host matrices
    //
    // A = local_rows x N
    // B = N x N
    // C = local_rows x N
    // --------------------------------------------------------

    std::vector<float> A(
        local_rows * N
    );

    std::vector<float> B(
        N * N
    );

    std::vector<float> C(
        local_rows * N,
        0.0f
    );

    // --------------------------------------------------------
    // Initialize A
    // --------------------------------------------------------

    for (size_t i = 0;
         i < A.size();
         ++i) {

        A[i] = 1.0f;
    }

    // --------------------------------------------------------
    // Initialize B
    // --------------------------------------------------------

    for (size_t i = 0;
         i < B.size();
         ++i) {

        B[i] = 2.0f;
    }

    // --------------------------------------------------------
    // Device memory
    // --------------------------------------------------------

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    try {

        d_A = sycl::malloc_device<float>(
            A.size(),
            q
        );

        d_B = sycl::malloc_device<float>(
            B.size(),
            q
        );

        d_C = sycl::malloc_device<float>(
            C.size(),
            q
        );

        if (!d_A || !d_B || !d_C) {

            std::cerr
                << "[Rank "
                << rank
                << "] Device memory allocation failed.\n";

            sycl::free(d_A, q);
            sycl::free(d_B, q);
            sycl::free(d_C, q);

            MPI_Abort(
                MPI_COMM_WORLD,
                1
            );
        }

        // ----------------------------------------------------
        // Copy data to device
        // ----------------------------------------------------

        q.memcpy(
            d_A,
            A.data(),
            A.size() * sizeof(float)
        );

        q.memcpy(
            d_B,
            B.data(),
            B.size() * sizeof(float)
        );

        q.wait();

        // ----------------------------------------------------
        // Synchronize all MPI ranks
        // ----------------------------------------------------

        MPI_Barrier(
            MPI_COMM_WORLD
        );

        if (rank == 0) {

            std::cout
                << "Starting matrix multiplication...\n";
        }

        // ----------------------------------------------------
        // Start timer
        // ----------------------------------------------------

        MPI_Barrier(
            MPI_COMM_WORLD
        );

        auto start =
            std::chrono::high_resolution_clock::now();

        // ----------------------------------------------------
        // Global dimensions
        //
        // Each rank calculates:
        //
        // local_rows x N
        //
        // of the final matrix.
        // ----------------------------------------------------

        const size_t global_rows =
            local_rows;

        const size_t global_cols =
            N;

        // ----------------------------------------------------
        // SYCL tiled matrix multiplication
        // ----------------------------------------------------

        q.submit(
            [&](sycl::handler &h) {

                sycl::local_accessor<float, 2> localA(
                    sycl::range<2>(
                        TILE,
                        TILE
                    ),
                    h
                );

                sycl::local_accessor<float, 2> localB(
                    sycl::range<2>(
                        TILE,
                        TILE
                    ),
                    h
                );

                sycl::range<2> global_range(
                    global_rows,
                    global_cols
                );

                sycl::range<2> local_range(
                    TILE,
                    TILE
                );

                h.parallel_for(
                    sycl::nd_range<2>(
                        global_range,
                        local_range
                    ),
                    [=](
                        sycl::nd_item<2> item
                    ) {

                        const size_t row =
                            item.get_global_id(0);

                        const size_t col =
                            item.get_global_id(1);

                        const size_t local_row =
                            item.get_local_id(0);

                        const size_t local_col =
                            item.get_local_id(1);

                        float sum = 0.0f;

                        // ------------------------------------------------
                        // Iterate over tiles
                        // ------------------------------------------------

                        for (
                            size_t tile = 0;
                            tile < N;
                            tile += TILE
                        ) {

                            // Load A tile

                            localA[
                                local_row,
                                local_col
                            ] =
                                d_A[
                                    row * N
                                    +
                                    tile
                                    +
                                    local_col
                                ];

                            // Load B tile

                            localB[
                                local_row,
                                local_col
                            ] =
                                d_B[
                                    (tile + local_row)
                                    * N
                                    +
                                    col
                                ];

                            // Synchronize work-group

                            item.barrier(
                                sycl::access::fence_space::local_space
                            );

                            // Compute tile

                            for (
                                size_t k = 0;
                                k < TILE;
                                ++k
                            ) {

                                sum +=
                                    localA[
                                        local_row,
                                        k
                                    ]
                                    *
                                    localB[
                                        k,
                                        local_col
                                    ];
                            }

                            // Synchronize before
                            // loading next tile

                            item.barrier(
                                sycl::access::fence_space::local_space
                            );
                        }

                        d_C[
                            row * N
                            +
                            col
                        ] = sum;
                    }
                );
            }
        ).wait();

        // ----------------------------------------------------
        // End timer
        // ----------------------------------------------------

        auto end =
            std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed =
            end - start;

        // ----------------------------------------------------
        // Copy result back
        // ----------------------------------------------------

        q.memcpy(
            C.data(),
            d_C,
            C.size() * sizeof(float)
        ).wait();

        // ----------------------------------------------------
        // Verify result
        // ----------------------------------------------------

        bool correct = true;

        const float expected =
            2.0f *
            static_cast<float>(N);

        for (size_t i = 0;
             i < C.size();
             ++i) {

            if (
                std::fabs(
                    C[i] - expected
                ) > 1e-3f
            ) {

                correct = false;
                break;
            }
        }

        // ----------------------------------------------------
        // Performance
        // ----------------------------------------------------

        double operations =
            2.0
            *
            static_cast<double>(local_rows)
            *
            static_cast<double>(N)
            *
            static_cast<double>(N);

        double gflops =
            operations
            /
            elapsed.count()
            /
            1.0e9;

        // ----------------------------------------------------
        // Rank result
        // ----------------------------------------------------

        std::cout
            << "\n[Rank "
            << rank
            << "] Results\n";

        std::cout
            << "----------------------------------------\n";

        std::cout
            << "[Rank "
            << rank
            << "] C[0]         : "
            << C[0]
            << '\n';

        std::cout
            << "[Rank "
            << rank
            << "] Verification : "
            << (
                correct
                ? "PASSED"
                : "FAILED"
            )
            << '\n';

        std::cout
            << "[Rank "
            << rank
            << "] Time         : "
            << elapsed.count()
            << " seconds\n";

        std::cout
            << "[Rank "
            << rank
            << "] GFLOPS       : "
            << gflops
            << '\n';

        std::cout
            << "----------------------------------------\n";

        // ----------------------------------------------------
        // Global synchronization
        // ----------------------------------------------------

        MPI_Barrier(
            MPI_COMM_WORLD
        );

        if (rank == 0) {

            std::cout
                << "\n========================================\n"
                << "             COMPLETED\n"
                << "========================================\n";
        }

        // ----------------------------------------------------
        // Cleanup
        // ----------------------------------------------------

        sycl::free(
            d_A,
            q
        );

        sycl::free(
            d_B,
            q
        );

        sycl::free(
            d_C,
            q
        );
    }
    catch (const sycl::exception &e) {

        std::cerr
            << "[Rank "
            << rank
            << "] SYCL exception:\n"
            << e.what()
            << '\n';

        MPI_Abort(
            MPI_COMM_WORLD,
            1
        );
    }

    MPI_Finalize();

    return 0;
}
