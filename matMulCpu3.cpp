#include <sycl/sycl.hpp>
#include <mpi.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>

int main(int argc, char **argv)
{
    /*
     * ------------------------------------------------------------
     * MPI INITIALIZATION
     * ------------------------------------------------------------
     */

    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    /*
     * ------------------------------------------------------------
     * MATRIX SIZE
     *
     * Default:
     *
     *     30000 x 30000
     *
     * You can also pass N from command line:
     *
     *     mpirun ./matMulMPI 10000
     * ------------------------------------------------------------
     */

    size_t N = 20000;

    if (argc > 1)
    {
        N = std::stoull(argv[1]);
    }


    /*
     * ------------------------------------------------------------
     * SYCL CPU QUEUE
     * ------------------------------------------------------------
     */

    sycl::queue q(
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
                        << "Rank asynchronous SYCL exception: "
                        << ex.what()
                        << std::endl;
                }
            }
        }
    );


    /*
     * ------------------------------------------------------------
     * DEVICE INFORMATION
     * ------------------------------------------------------------
     */

    auto device = q.get_device();

    if (rank == 0)
    {
        std::cout
            << "========================================\n";

        std::cout
            << "       MPI + SYCL MULTI-NODE CPU\n";

        std::cout
            << "       MATRIX MULTIPLICATION\n";

        std::cout
            << "========================================\n";

        std::cout
            << "\nMPI processes : "
            << size
            << '\n';

        std::cout
            << "Matrix size   : "
            << N
            << " x "
            << N
            << '\n';
    }

    std::cout
        << "Rank "
        << rank
        << " running on SYCL device: "
        << device.get_info<
            sycl::info::device::name>()
        << '\n';


    /*
     * ------------------------------------------------------------
     * DISTRIBUTE MATRIX ROWS
     *
     * Example:
     *
     * N = 100
     * MPI = 4
     *
     * Rank 0 -> rows 0-24
     * Rank 1 -> rows 25-49
     * Rank 2 -> rows 50-74
     * Rank 3 -> rows 75-99
     * ------------------------------------------------------------
     */

    const size_t base_rows = N / size;

    const size_t remainder = N % size;

    const size_t local_rows =
        base_rows +
        (static_cast<size_t>(rank) < remainder ? 1 : 0);

    /*
     * Starting row of this MPI rank.
     */

    const size_t start_row =
        rank * base_rows +
        std::min(
            static_cast<size_t>(rank),
            remainder
        );


    /*
     * ------------------------------------------------------------
     * MATRIX B
     *
     * Every MPI rank needs B.
     *
     * B is initialized identically on every rank.
     *
     * This avoids sending a ~3.35 GB matrix through MPI.
     * ------------------------------------------------------------
     */

    const size_t total_elements = N * N;

    std::vector<float> B(total_elements);

    for (size_t i = 0; i < total_elements; ++i)
    {
        B[i] = 2.0f;
    }


    /*
     * ------------------------------------------------------------
     * LOCAL MATRIX A
     *
     * Each rank only stores its assigned rows.
     * ------------------------------------------------------------
     */

    const size_t local_elements =
        local_rows * N;

    std::vector<float> A(local_elements);

    for (size_t i = 0; i < local_elements; ++i)
    {
        A[i] = 1.0f;
    }


    /*
     * ------------------------------------------------------------
     * LOCAL RESULT
     *
     * Rank only calculates its own rows.
     * ------------------------------------------------------------
     */

    std::vector<float> C(local_elements, 0.0f);


    /*
     * ------------------------------------------------------------
     * ALLOCATE SYCL SHARED MEMORY
     *
     * Shared USM works with the CPU SYCL backend.
     * ------------------------------------------------------------
     */

    float *d_A =
        sycl::malloc_shared<float>(
            local_elements,
            q
        );

    float *d_B =
        sycl::malloc_shared<float>(
            total_elements,
            q
        );

    float *d_C =
        sycl::malloc_shared<float>(
            local_elements,
            q
        );


    if (!d_A || !d_B || !d_C)
    {
        std::cerr
            << "Rank "
            << rank
            << ": SYCL memory allocation failed."
            << std::endl;

        MPI_Abort(
            MPI_COMM_WORLD,
            1
        );
    }


    /*
     * ------------------------------------------------------------
     * COPY DATA TO SYCL MEMORY
     * ------------------------------------------------------------
     */

    for (size_t i = 0; i < local_elements; ++i)
    {
        d_A[i] = A[i];
        d_C[i] = 0.0f;
    }

    for (size_t i = 0; i < total_elements; ++i)
    {
        d_B[i] = B[i];
    }


    /*
     * ------------------------------------------------------------
     * TILE SIZE
     * ------------------------------------------------------------
     */

    constexpr size_t TILE = 16;


    /*
     * ------------------------------------------------------------
     * START GLOBAL COMPUTATION TIMER
     * ------------------------------------------------------------
     */

    MPI_Barrier(MPI_COMM_WORLD);

    double start_time =
        MPI_Wtime();


    /*
     * ------------------------------------------------------------
     * SYCL MATRIX MULTIPLICATION
     *
     * Each MPI rank computes:
     *
     *     C_local = A_local × B
     *
     * ------------------------------------------------------------
     */

    q.submit(
        [&](sycl::handler &h)
        {
            /*
             * Local tile for A.
             *
             * 1D accessor avoids the accessor indexing
             * problem encountered with your compiler.
             */

            sycl::local_accessor<float, 1> localA(
                sycl::range<1>(
                    TILE * TILE
                ),
                h
            );


            /*
             * Local tile for B.
             */

            sycl::local_accessor<float, 1> localB(
                sycl::range<1>(
                    TILE * TILE
                ),
                h
            );


            /*
             * Round local rows and N up to TILE.
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
                        local_row * TILE +
                        local_col;


                    float sum = 0.0f;


                    /*
                     * Process B in tiles.
                     */

                    for (
                        size_t tile = 0;
                        tile < N;
                        tile += TILE
                    )
                    {

                        /*
                         * ------------------------------------------------
                         * LOAD A TILE
                         * ------------------------------------------------
                         */

                        const size_t a_col =
                            tile + local_col;


                        if (
                            row < local_rows &&
                            a_col < N
                        )
                        {
                            localA[local_index] =
                                d_A[
                                    row * N +
                                    a_col
                                ];
                        }
                        else
                        {
                            localA[local_index] =
                                0.0f;
                        }


                        /*
                         * ------------------------------------------------
                         * LOAD B TILE
                         * ------------------------------------------------
                         */

                        const size_t b_row =
                            tile + local_row;


                        if (
                            b_row < N &&
                            col < N
                        )
                        {
                            localB[local_index] =
                                d_B[
                                    b_row * N +
                                    col
                                ];
                        }
                        else
                        {
                            localB[local_index] =
                                0.0f;
                        }


                        /*
                         * Wait until entire tile
                         * is loaded.
                         */

                        item.barrier(
                            sycl::access::fence_space::local_space
                        );


                        /*
                         * ------------------------------------------------
                         * MULTIPLY TILE
                         * ------------------------------------------------
                         */

                        for (
                            size_t k = 0;
                            k < TILE;
                            ++k
                        )
                        {
                            const float a =
                                localA[
                                    local_row *
                                    TILE +
                                    k
                                ];


                            const float b =
                                localB[
                                    k *
                                    TILE +
                                    local_col
                                ];


                            sum += a * b;
                        }


                        /*
                         * Wait before loading
                         * next tile.
                         */

                        item.barrier(
                            sycl::access::fence_space::local_space
                        );
                    }


                    /*
                     * ------------------------------------------------
                     * STORE RESULT
                     * ------------------------------------------------
                     */

                    if (
                        row < local_rows &&
                        col < N
                    )
                    {
                        d_C[
                            row * N +
                            col
                        ] = sum;
                    }
                }
            );
        }
    ).wait();


    /*
     * ------------------------------------------------------------
     * STOP TIMER
     * ------------------------------------------------------------
     */

    MPI_Barrier(MPI_COMM_WORLD);

    double end_time =
        MPI_Wtime();


    /*
     * ------------------------------------------------------------
     * COPY RESULT BACK
     * ------------------------------------------------------------
     */

    for (size_t i = 0; i < local_elements; ++i)
    {
        C[i] = d_C[i];
    }


    /*
     * ------------------------------------------------------------
     * VERIFY LOCAL RESULT
     * ------------------------------------------------------------
     */

    bool local_correct = true;

    const float expected =
        2.0f *
        static_cast<float>(N);


    for (size_t i = 0; i < local_elements; ++i)
    {
        if (std::fabs(C[i] - expected) > 0.001f)
        {
            local_correct = false;
            break;
        }
    }


    /*
     * ------------------------------------------------------------
     * CHECK ALL MPI RANKS
     * ------------------------------------------------------------
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
     * ------------------------------------------------------------
     * PERFORMANCE
     * ------------------------------------------------------------
     */

    double max_time = 0.0;

    MPI_Reduce(
        &end_time,
        &max_time,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );


    if (rank == 0)
    {
        const double operations =
            2.0 *
            static_cast<double>(N) *
            static_cast<double>(N) *
            static_cast<double>(N);


        const double gflops =
            operations /
            max_time /
            1.0e9;


        std::cout
            << "\n========================================\n";

        std::cout
            << "              RESULTS\n";

        std::cout
            << "========================================\n";


        std::cout
            << std::fixed
            << std::setprecision(2);


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
            << "========================================\n";
    }


    /*
     * ------------------------------------------------------------
     * FREE SYCL MEMORY
     * ------------------------------------------------------------
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


    MPI_Finalize();

    return 0;
}
