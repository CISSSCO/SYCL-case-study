#include <sycl/sycl.hpp>
#include <mpi.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <algorithm>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Matrix size
    const int N = (argc > 1) ? std::atoi(argv[1]) : 4096;

    // Make sure N can be divided among MPI ranks
    if (N % size != 0)
    {
        if (rank == 0)
        {
            std::cerr
                << "Error: N must be divisible by number of MPI ranks.\n"
                << "N = " << N
                << ", MPI ranks = " << size << '\n';
        }

        MPI_Finalize();
        return 1;
    }

    const int rows_per_rank = N / size;

    try
    {
        /*
         * Select CPU or GPU automatically.
         *
         * On CPU nodes:
         *     SYCL_CPU=1
         *
         * On GPU nodes:
         *     SYCL_CPU=0
         */
        const char *cpu_env = std::getenv("SYCL_CPU");

        bool use_cpu =
            cpu_env && std::string(cpu_env) == "1";

        sycl::queue q;

        if (use_cpu)
        {
            q = sycl::queue(
                sycl::cpu_selector_v,
                [](sycl::exception_list exceptions)
                {
                    for (auto &e : exceptions)
                    {
                        try
                        {
                            std::rethrow_exception(e);
                        }
                        catch (const sycl::exception &ex)
                        {
                            std::cerr
                                << "Asynchronous SYCL exception: "
                                << ex.what() << '\n';
                        }
                    }
                });
        }
        else
        {
            q = sycl::queue(
                sycl::gpu_selector_v,
                [](sycl::exception_list exceptions)
                {
                    for (auto &e : exceptions)
                    {
                        try
                        {
                            std::rethrow_exception(e);
                        }
                        catch (const sycl::exception &ex)
                        {
                            std::cerr
                                << "Asynchronous SYCL exception: "
                                << ex.what() << '\n';
                        }
                    }
                });
        }

        auto device = q.get_device();

        if (rank == 0)
        {
            std::cout
                << "========================================\n"
                << "       MPI + SYCL MATRIX MULTIPLY\n"
                << "========================================\n";

            std::cout
                << "Matrix size     : "
                << N << " x " << N << '\n';

            std::cout
                << "MPI ranks       : "
                << size << '\n';

            std::cout
                << "Rows per rank   : "
                << rows_per_rank << '\n';

            std::cout
                << "Execution model : "
                << (use_cpu ? "CPU" : "GPU") << '\n';

            std::cout
                << "========================================\n";
        }

        std::cout
            << "[Rank " << rank << "] "
            << "Host: " << []()
            {
                char hostname[256];
                gethostname(hostname, sizeof(hostname));
                return std::string(hostname);
            }()
            << " | Device: "
            << device.get_info<
                sycl::info::device::name>()
            << '\n';

        /*
         * A is distributed by rows.
         *
         * Each rank owns:
         *
         * rows_per_rank x N
         *
         * B is replicated on every rank.
         *
         * C is distributed by rows.
         */

        const size_t local_elements =
            static_cast<size_t>(rows_per_rank) * N;

        const size_t matrix_elements =
            static_cast<size_t>(N) * N;

        std::vector<float> local_A(local_elements);
        std::vector<float> B(matrix_elements);
        std::vector<float> local_C(local_elements, 0.0f);

        /*
         * Initialize local A.
         */
        for (size_t i = 0; i < local_elements; ++i)
        {
            local_A[i] = 1.0f;
        }

        /*
         * Initialize B.
         */
        for (size_t i = 0; i < matrix_elements; ++i)
        {
            B[i] = 2.0f;
        }

        /*
         * Allocate SYCL USM memory.
         */
        float *d_A =
            sycl::malloc_device<float>(
                local_elements, q);

        float *d_B =
            sycl::malloc_device<float>(
                matrix_elements, q);

        float *d_C =
            sycl::malloc_device<float>(
                local_elements, q);

        if (!d_A || !d_B || !d_C)
        {
            std::cerr
                << "[Rank " << rank
                << "] Device memory allocation failed.\n";

            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /*
         * Copy data to device.
         */
        q.memcpy(
            d_A,
            local_A.data(),
            local_elements * sizeof(float));

        q.memcpy(
            d_B,
            B.data(),
            matrix_elements * sizeof(float));

        q.memset(
            d_C,
            0,
            local_elements * sizeof(float));

        q.wait();

        /*
         * Synchronize all MPI ranks before timing.
         */
        MPI_Barrier(MPI_COMM_WORLD);

        double start =
            MPI_Wtime();

        /*
         * ------------------------------------------------
         * TILED MATRIX MULTIPLICATION
         * ------------------------------------------------
         */

        constexpr int TILE = 16;

        sycl::range<2> global(
            ((rows_per_rank + TILE - 1) / TILE) * TILE,
            ((N + TILE - 1) / TILE) * TILE);

        sycl::range<2> local(
            TILE,
            TILE);

        /*
         * Local accessor for A and B tiles.
         */
        sycl::local_accessor<float, 2> tileA(
            sycl::range<2>(TILE, TILE));

        sycl::local_accessor<float, 2> tileB(
            sycl::range<2>(TILE, TILE));

        q.submit(
            [&](sycl::handler &h)
            {
                h.parallel_for(
                    sycl::nd_range<2>(
                        global,
                        local),
                    tileA,
                    tileB,
                    [=](sycl::nd_item<2> item,
                        auto A_tile,
                        auto B_tile)
                    {
                        const int row =
                            item.get_group(0) * TILE +
                            item.get_local_id(0);

                        const int col =
                            item.get_group(1) * TILE +
                            item.get_local_id(1);

                        float sum = 0.0f;

                        const int local_row =
                            item.get_local_id(0);

                        const int local_col =
                            item.get_local_id(1);

                        for (
                            int tile = 0;
                            tile < N;
                            tile += TILE)
                        {
                            /*
                             * Load A tile.
                             */
                            if (row < rows_per_rank &&
                                tile + local_col < N)
                            {
                                A_tile[local_row][local_col] =
                                    d_A[
                                        row * N +
                                        tile +
                                        local_col
                                    ];
                            }
                            else
                            {
                                A_tile[local_row][local_col] =
                                    0.0f;
                            }

                            /*
                             * Load B tile.
                             */
                            if (tile + local_row < N &&
                                col < N)
                            {
                                B_tile[local_row][local_col] =
                                    d_B[
                                        (tile + local_row) * N +
                                        col
                                    ];
                            }
                            else
                            {
                                B_tile[local_row][local_col] =
                                    0.0f;
                            }

                            item.barrier(
                                sycl::access::fence_space::local_space);

                            /*
                             * Multiply tiles.
                             */
                            for (int k = 0;
                                 k < TILE;
                                 ++k)
                            {
                                sum +=
                                    A_tile[local_row][k] *
                                    B_tile[k][local_col];
                            }

                            item.barrier(
                                sycl::access::fence_space::local_space);
                        }

                        /*
                         * Store result.
                         */
                        if (row < rows_per_rank &&
                            col < N)
                        {
                            d_C[
                                row * N +
                                col
                            ] = sum;
                        }
                    });
            });

        q.wait();

        double end =
            MPI_Wtime();

        double local_time =
            end - start;

        /*
         * Get maximum execution time across ranks.
         *
         * The slowest rank determines total
         * distributed execution time.
         */
        double total_time = 0.0;

        MPI_Reduce(
            &local_time,
            &total_time,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD);

        /*
         * Copy result back.
         */
        q.memcpy(
            local_C.data(),
            d_C,
            local_elements * sizeof(float))
            .wait();

        /*
         * Verify local result.
         *
         * A = 1
         * B = 2
         *
         * Therefore:
         *
         * C[i][j] = 2*N
         */
        bool local_correct = true;

        const float expected =
            2.0f * static_cast<float>(N);

        for (size_t i = 0;
             i < local_elements;
             ++i)
        {
            if (std::fabs(local_C[i] - expected)
                > 1e-3f)
            {
                local_correct = false;
                break;
            }
        }

        int local_status =
            local_correct ? 1 : 0;

        int global_status = 0;

        MPI_Reduce(
            &local_status,
            &global_status,
            1,
            MPI_INT,
            MPI_MIN,
            0,
            MPI_COMM_WORLD);

        /*
         * Calculate GFLOPS.
         */
        double operations =
            2.0 *
            static_cast<double>(N) *
            static_cast<double>(N) *
            static_cast<double>(N);

        if (rank == 0)
        {
            double gflops =
                operations /
                total_time /
                1.0e9;

            std::cout
                << "\n========================================\n"
                << "               RESULTS\n"
                << "========================================\n";

            std::cout
                << "Matrix size     : "
                << N << " x " << N << '\n';

            std::cout
                << "MPI ranks       : "
                << size << '\n';

            std::cout
                << "Execution time  : "
                << total_time
                << " seconds\n";

            std::cout
                << "Performance     : "
                << gflops
                << " GFLOPS\n";

            std::cout
                << "Verification    : "
                << (global_status ? "PASSED" : "FAILED")
                << '\n';

            std::cout
                << "========================================\n";
        }

        /*
         * Free device memory.
         */
        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_C, q);
    }
    catch (const sycl::exception &e)
    {
        std::cerr
            << "[Rank " << rank
            << "] SYCL exception:\n"
            << e.what()
            << '\n';

        MPI_Abort(
            MPI_COMM_WORLD,
            1);
    }

    MPI_Finalize();

    return 0;
}
