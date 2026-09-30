#include <sycl/sycl.hpp>
#include <mpi.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>


int main(int argc, char **argv)
{
    /*
     * ============================================================
     * MPI INITIALIZATION
     * ============================================================
     */

    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    /*
     * ============================================================
     * GET LOCAL MPI INFORMATION
     *
     * We expect:
     *
     *     2 MPI ranks per node
     *
     * Each rank will use one GPU.
     * ============================================================
     */

    MPI_Comm local_comm;

    MPI_Comm_split_type(
        MPI_COMM_WORLD,
        MPI_COMM_TYPE_SHARED,
        rank,
        MPI_INFO_NULL,
        &local_comm
    );

    int local_rank = 0;
    int local_size = 1;

    MPI_Comm_rank(
        local_comm,
        &local_rank
    );

    MPI_Comm_size(
        local_comm,
        &local_size
    );


    /*
     * ============================================================
     * MATRIX SIZE
     *
     * Default:
     *
     *     20000 x 20000
     *
     * Example:
     *
     *     srun ./matmul_mpi_sycl_gpu 5000
     * ============================================================
     */

    size_t N = 20000;

    if (argc > 1)
    {
        N = std::stoull(argv[1]);
    }


    /*
     * ============================================================
     * GPU DEVICE DISCOVERY
     * ============================================================
     */

    auto platforms =
        sycl::platform::get_platforms();

    std::vector<sycl::device> gpu_devices;

    for (const auto &platform : platforms)
    {
        for (const auto &device :
             platform.get_devices(
                 sycl::info::device_type::gpu))
        {
            gpu_devices.push_back(device);
        }
    }


    /*
     * ============================================================
     * CHECK GPU AVAILABILITY
     * ============================================================
     */

    if (gpu_devices.empty())
    {
        std::cerr
            << "Rank "
            << rank
            << ": No SYCL GPU devices found.\n";

        MPI_Abort(
            MPI_COMM_WORLD,
            1
        );
    }


    /*
     * ============================================================
     * SELECT GPU
     *
     * local_rank determines which GPU is used.
     *
     * With 2 GPUs per node:
     *
     * local_rank 0 -> GPU 0
     * local_rank 1 -> GPU 1
     *
     * ============================================================
     */

    const size_t gpu_id =
        static_cast<size_t>(local_rank)
        % gpu_devices.size();

    sycl::device selected_device =
        gpu_devices[gpu_id];


    /*
     * ============================================================
     * CREATE SYCL QUEUE
     * ============================================================
     */

    sycl::queue q(
        selected_device,
        [](sycl::exception_list exceptions)
        {
            for (const auto &e : exceptions)
            {
                try
                {
                    std::rethrow_exception(e);
                }
                catch (const sycl::exception &ex)
                {
                    std::cerr
                        << "SYCL asynchronous exception: "
                        << ex.what()
                        << '\n';
                }
            }
        }
    );


    /*
     * ============================================================
     * DEVICE INFORMATION
     * ============================================================
     */

    if (rank == 0)
    {
        std::cout
            << "============================================\n"
            << "       MPI + SYCL MULTI-NODE GPU\n"
            << "       MATRIX MULTIPLICATION\n"
            << "============================================\n\n";

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
            << "MPI ranks/node: "
            << local_size
            << '\n';

        std::cout
            << "GPUs/rank     : 1\n\n";
    }


    std::cout
        << "Rank "
        << rank
        << " | Local rank "
        << local_rank
        << " | GPU "
        << gpu_id
        << " | "
        << selected_device.get_info<
               sycl::info::device::name>()
        << '\n';

    MPI_Barrier(MPI_COMM_WORLD);


    /*
     * ============================================================
     * DISTRIBUTE MATRIX A ROWS
     * ============================================================
     */

    const size_t base_rows =
        N / size;

    const size_t remainder =
        N % size;


    /*
     * Number of rows assigned to this rank.
     */

    const size_t local_rows =
        base_rows +
        (
            static_cast<size_t>(rank)
            < remainder
                ? 1
                : 0
        );


    /*
     * Global starting row.
     */

    const size_t start_row =
        rank * base_rows
        +
        std::min(
            static_cast<size_t>(rank),
            remainder
        );


    /*
     * ============================================================
     * MATRIX SIZES
     * ============================================================
     */

    const size_t local_elements =
        local_rows * N;

    const size_t total_elements =
        N * N;


    /*
     * ============================================================
     * MATRIX B
     *
     * Every MPI rank receives its own copy.
     *
     * B[i][j] = 2
     * ============================================================
     */

    std::vector<float> B(
        total_elements,
        2.0f
    );


    /*
     * ============================================================
     * LOCAL MATRIX A
     *
     * Each rank stores only its assigned rows.
     *
     * A[i][j] = 1
     * ============================================================
     */

    std::vector<float> A(
        local_elements
    );

    for (size_t row = 0;
         row < local_rows;
         ++row)
    {
        const size_t global_row =
            start_row + row;

        for (size_t col = 0;
             col < N;
             ++col)
        {
            A[row * N + col] =
                1.0f;
        }

        (void)global_row;
    }


    /*
     * ============================================================
     * LOCAL RESULT
     * ============================================================
     */

    std::vector<float> C(
        local_elements,
        0.0f
    );


    /*
     * ============================================================
     * SYCL DEVICE USM
     *
     * IMPORTANT:
     *
     * Unlike the CPU version, this memory is allocated on
     * the selected GPU.
     * ============================================================
     */

    float *d_A =
        sycl::malloc_device<float>(
            local_elements,
            q
        );

    float *d_B =
        sycl::malloc_device<float>(
            total_elements,
            q
        );

    float *d_C =
        sycl::malloc_device<float>(
            local_elements,
            q
        );


    /*
     * Check allocations.
     */

    if (!d_A || !d_B || !d_C)
    {
        std::cerr
            << "Rank "
            << rank
            << ": GPU memory allocation failed.\n";

        MPI_Abort(
            MPI_COMM_WORLD,
            1
        );
    }


    /*
     * ============================================================
     * COPY HOST -> GPU
     * ============================================================
     */

    q.memcpy(
        d_A,
        A.data(),
        local_elements * sizeof(float)
    );

    q.memcpy(
        d_B,
        B.data(),
        total_elements * sizeof(float)
    );

    q.memset(
        d_C,
        0,
        local_elements * sizeof(float)
    );

    q.wait();


    /*
     * ============================================================
     * TILE SIZE
     * ============================================================
     */

    constexpr size_t TILE = 16;


    /*
     * ============================================================
     * SYNCHRONIZE MPI PROCESSES
     * ============================================================
     */

    MPI_Barrier(
        MPI_COMM_WORLD
    );

    const double start_time =
        MPI_Wtime();


    /*
     * ============================================================
     * SYCL GPU MATRIX MULTIPLICATION
     *
     * Each MPI rank calculates:
     *
     *     C_local = A_local × B
     *
     * on its assigned GPU.
     * ============================================================
     */

    q.submit(
        [&](sycl::handler &h)
        {
            /*
             * Shared/local memory tiles.
             */

            sycl::local_accessor<float, 1> localA(
                sycl::range<1>(
                    TILE * TILE
                ),
                h
            );

            sycl::local_accessor<float, 1> localB(
                sycl::range<1>(
                    TILE * TILE
                ),
                h
            );


            /*
             * Round dimensions to TILE.
             */

            const size_t global_rows =
                (
                    (local_rows + TILE - 1)
                    / TILE
                ) * TILE;

            const size_t global_cols =
                (
                    (N + TILE - 1)
                    / TILE
                ) * TILE;


            sycl::nd_range<2> execution_range(
                sycl::range<2>(
                    global_rows,
                    global_cols
                ),
                sycl::range<2>(
                    TILE,
                    TILE
                )
            );


            /*
             * ====================================================
             * GPU KERNEL
             * ====================================================
             */

            h.parallel_for(
                execution_range,
                [=](sycl::nd_item<2> item)
                {
                    const size_t local_row =
                        item.get_local_id(0);

                    const size_t local_col =
                        item.get_local_id(1);

                    const size_t row =
                        item.get_global_id(0);

                    const size_t col =
                        item.get_global_id(1);


                    const size_t local_index =
                        local_row * TILE
                        + local_col;


                    float sum = 0.0f;


                    /*
                     * Process B in tiles.
                     */

                    for (
                        size_t tile_start = 0;
                        tile_start < N;
                        tile_start += TILE
                    )
                    {

                        /*
                         * ----------------------------------------
                         * LOAD A TILE
                         * ----------------------------------------
                         */

                        const size_t a_col =
                            tile_start
                            + local_col;

                        if (
                            row < local_rows &&
                            a_col < N
                        )
                        {
                            localA[local_index] =
                                d_A[
                                    row * N
                                    + a_col
                                ];
                        }
                        else
                        {
                            localA[local_index] =
                                0.0f;
                        }


                        /*
                         * ----------------------------------------
                         * LOAD B TILE
                         * ----------------------------------------
                         */

                        const size_t b_row =
                            tile_start
                            + local_row;

                        if (
                            b_row < N &&
                            col < N
                        )
                        {
                            localB[local_index] =
                                d_B[
                                    b_row * N
                                    + col
                                ];
                        }
                        else
                        {
                            localB[local_index] =
                                0.0f;
                        }


                        /*
                         * Wait until the complete tile
                         * has been loaded.
                         */

                        item.barrier(
                            sycl::access::fence_space::local_space
                        );


                        /*
                         * ----------------------------------------
                         * COMPUTE TILE
                         * ----------------------------------------
                         */

                        for (
                            size_t k = 0;
                            k < TILE;
                            ++k
                        )
                        {
                            sum +=
                                localA[
                                    local_row * TILE
                                    + k
                                ]
                                *
                                localB[
                                    k * TILE
                                    + local_col
                                ];
                        }


                        /*
                         * Wait before loading
                         * the next tile.
                         */

                        item.barrier(
                            sycl::access::fence_space::local_space
                        );
                    }


                    /*
                     * ----------------------------------------
                     * STORE RESULT
                     * ----------------------------------------
                     */

                    if (
                        row < local_rows &&
                        col < N
                    )
                    {
                        d_C[
                            row * N + col
                        ] = sum;
                    }
                }
            );
        }
    ).wait();


    /*
     * ============================================================
     * STOP TIMER
     * ============================================================
     */

    MPI_Barrier(
        MPI_COMM_WORLD
    );

    const double end_time =
        MPI_Wtime();

    const double local_time =
        end_time - start_time;


    /*
     * ============================================================
     * COPY GPU RESULT -> HOST
     * ============================================================
     */

    q.memcpy(
        C.data(),
        d_C,
        local_elements * sizeof(float)
    ).wait();


    /*
     * ============================================================
     * VERIFY RESULT
     *
     * A = 1
     * B = 2
     *
     * Therefore:
     *
     * C[i][j] = 2 * N
     * ============================================================
     */

    const float expected =
        2.0f
        * static_cast<float>(N);

    bool local_correct =
        true;

    for (size_t i = 0;
         i < local_elements;
         ++i)
    {
        if (
            std::fabs(
                C[i] - expected
            ) > 0.001f
        )
        {
            local_correct = false;
            break;
        }
    }


    /*
     * ============================================================
     * GLOBAL VERIFICATION
     * ============================================================
     */

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
        MPI_COMM_WORLD
    );


    /*
     * ============================================================
     * GLOBAL EXECUTION TIME
     *
     * Slowest MPI rank determines total runtime.
     * ============================================================
     */

    double max_time = 0.0;

    MPI_Reduce(
        &local_time,
        &max_time,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );


    /*
     * ============================================================
     * PERFORMANCE
     * ============================================================
     */

    if (rank == 0)
    {
        const double operations =
            2.0
            * static_cast<double>(N)
            * static_cast<double>(N)
            * static_cast<double>(N);


        const double gflops =
            operations
            / max_time
            / 1.0e9;


        /*
         * ========================================================
         * RESULTS
         * ========================================================
         */

        std::cout
            << "\n============================================\n"
            << "                 RESULTS\n"
            << "============================================\n";

        std::cout
            << std::fixed
            << std::setprecision(2);

        std::cout
            << "MPI processes : "
            << size
            << '\n';

        std::cout
            << "MPI ranks/node: "
            << local_size
            << '\n';

        std::cout
            << "Total GPUs    : "
            << size
            << '\n';

        std::cout
            << "Matrix size   : "
            << N
            << " x "
            << N
            << '\n';

        std::cout
            << "Execution time: "
            << max_time
            << " seconds\n";

        std::cout
            << "Performance   : "
            << gflops
            << " GFLOPS\n";

        std::cout
            << "Verification  : "
            << (
                global_status
                    ? "PASSED"
                    : "FAILED"
            )
            << '\n';

        std::cout
            << "============================================\n";
    }


    /*
     * ============================================================
     * FREE GPU MEMORY
     * ============================================================
     */

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


    /*
     * ============================================================
     * FREE LOCAL MPI COMMUNICATOR
     * ============================================================
     */

    MPI_Comm_free(
        &local_comm
    );


    /*
     * ============================================================
     * MPI FINALIZATION
     * ============================================================
     */

    MPI_Finalize();

    return 0;
}
