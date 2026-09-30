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
     * MATRIX SIZE
     *
     * Default:
     *
     *     20000 x 20000
     *
     * Optional:
     *
     *     srun ./matmul_mpi_sycl_cpu 5000
     * ============================================================
     */

    size_t N = 20000;

    if (argc > 1)
    {
        N = std::stoull(argv[1]);
    }


    /*
     * ============================================================
     * SYCL CPU QUEUE
     * ============================================================
     */

    sycl::queue q(
        sycl::cpu_selector_v,
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

    const auto device = q.get_device();

    if (rank == 0)
    {
        std::cout
            << "============================================\n"
            << "       MPI + SYCL MULTI-NODE CPU\n"
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
            << "SYCL backend  : CPU\n\n";
    }

    /*
     * Print the device used by every MPI rank.
     */

    std::cout
        << "Rank "
        << rank
        << " -> "
        << device.get_info<
               sycl::info::device::name>()
        << '\n';

    MPI_Barrier(MPI_COMM_WORLD);


    /*
     * ============================================================
     * DISTRIBUTE MATRIX A ROWS
     *
     * Example:
     *
     * N = 100
     * MPI processes = 4
     *
     * Rank 0 -> rows 0-24
     * Rank 1 -> rows 25-49
     * Rank 2 -> rows 50-74
     * Rank 3 -> rows 75-99
     *
     * If N is not evenly divisible, the first few ranks
     * receive one additional row.
     * ============================================================
     */

    const size_t base_rows = N / size;

    const size_t remainder = N % size;

    const size_t local_rows =
        base_rows +
        (
            static_cast<size_t>(rank) < remainder
                ? 1
                : 0
        );


    /*
     * Global starting row belonging to this rank.
     */

    const size_t start_row =
        rank * base_rows +
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
     * HOST MATRIX B
     *
     * B is replicated on every MPI rank.
     *
     * Every element:
     *
     *     B[i][j] = 2
     *
     * This avoids broadcasting a large matrix using MPI.
     * ============================================================
     */

    std::vector<float> B(total_elements, 2.0f);


    /*
     * ============================================================
     * LOCAL MATRIX A
     *
     * Each rank stores only its assigned rows.
     *
     * Every element:
     *
     *     A[i][j] = 1
     *
     * start_row is used to make the global row ownership explicit.
     * ============================================================
     */

    std::vector<float> A(local_elements);

    for (size_t local_row = 0; local_row < local_rows; ++local_row)
    {
        const size_t global_row =
            start_row + local_row;

        for (size_t col = 0; col < N; ++col)
        {
            /*
             * The value is intentionally independent of the row.
             *
             * This keeps verification simple:
             *
             * C[i][j] = 2 * N
             */

            A[local_row * N + col] = 1.0f;
        }

        /*
         * Avoid unused-variable warnings while making
         * the global ownership explicit.
         */

        (void)global_row;
    }


    /*
     * ============================================================
     * LOCAL RESULT MATRIX
     *
     * Only the rows belonging to this MPI rank are calculated.
     * ============================================================
     */

    std::vector<float> C(
        local_elements,
        0.0f
    );


    /*
     * ============================================================
     * SYCL USM SHARED MEMORY
     *
     * Because we are using the SYCL CPU backend,
     * shared USM is suitable for this example.
     * ============================================================
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


    /*
     * Check memory allocation.
     */

    if (!d_A || !d_B || !d_C)
    {
        std::cerr
            << "Rank "
            << rank
            << ": SYCL memory allocation failed.\n";

        MPI_Abort(
            MPI_COMM_WORLD,
            1
        );
    }


    /*
     * ============================================================
     * COPY HOST DATA TO SYCL USM
     * ============================================================
     */

    std::copy(
        A.begin(),
        A.end(),
        d_A
    );

    std::copy(
        B.begin(),
        B.end(),
        d_B
    );

    std::fill(
        d_C,
        d_C + local_elements,
        0.0f
    );


    /*
     * ============================================================
     * TILE SIZE
     * ============================================================
     */

    constexpr size_t TILE = 16;


    /*
     * ============================================================
     * SYNCHRONIZE MPI RANKS
     *
     * All ranks start the computation together.
     * ============================================================
     */

    MPI_Barrier(MPI_COMM_WORLD);

    const double start_time =
        MPI_Wtime();


    /*
     * ============================================================
     * SYCL MATRIX MULTIPLICATION
     *
     * Each MPI rank calculates:
     *
     *     C_local = A_local × B
     *
     * Therefore:
     *
     *     Rank 0 -> its rows of C
     *     Rank 1 -> its rows of C
     *     ...
     * ============================================================
     */

    q.submit(
        [&](sycl::handler &h)
        {
            /*
             * Local tile for matrix A.
             */

            sycl::local_accessor<float, 1> localA(
                sycl::range<1>(
                    TILE * TILE
                ),
                h
            );


            /*
             * Local tile for matrix B.
             */

            sycl::local_accessor<float, 1> localB(
                sycl::range<1>(
                    TILE * TILE
                ),
                h
            );


            /*
             * Round dimensions up to TILE.
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


            /*
             * SYCL ND-range.
             */

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
             * SYCL kernel.
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
                        local_row * TILE +
                        local_col;


                    float sum = 0.0f;


                    /*
                     * Process matrices in TILE x TILE blocks.
                     */

                    for (
                        size_t tile_start = 0;
                        tile_start < N;
                        tile_start += TILE
                    )
                    {

                        /*
                         * --------------------------------------------
                         * LOAD A TILE
                         * --------------------------------------------
                         */

                        const size_t a_col =
                            tile_start + local_col;

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
                         * --------------------------------------------
                         * LOAD B TILE
                         * --------------------------------------------
                         */

                        const size_t b_row =
                            tile_start + local_row;

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
                         * Make sure the entire tile has
                         * been loaded before computation.
                         */

                        item.barrier(
                            sycl::access::fence_space::local_space
                        );


                        /*
                         * --------------------------------------------
                         * MULTIPLY TILE
                         * --------------------------------------------
                         */

                        for (
                            size_t k = 0;
                            k < TILE;
                            ++k
                        )
                        {
                            sum +=
                                localA[
                                    local_row * TILE + k
                                ]
                                *
                                localB[
                                    k * TILE + local_col
                                ];
                        }


                        /*
                         * Make sure all work-items have
                         * finished using the tile before
                         * loading the next tile.
                         */

                        item.barrier(
                            sycl::access::fence_space::local_space
                        );
                    }


                    /*
                     * --------------------------------------------
                     * STORE RESULT
                     * --------------------------------------------
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

    MPI_Barrier(MPI_COMM_WORLD);

    const double end_time =
        MPI_Wtime();

    const double local_time =
        end_time - start_time;


    /*
     * ============================================================
     * COPY RESULT BACK TO HOST
     * ============================================================
     */

    std::copy(
        d_C,
        d_C + local_elements,
        C.begin()
    );


    /*
     * ============================================================
     * VERIFY RESULT
     *
     * Since:
     *
     *     A = 1
     *     B = 2
     *
     * Therefore:
     *
     *     C[i][j] = 2 * N
     * ============================================================
     */

    const float expected =
        2.0f * static_cast<float>(N);

    bool local_correct = true;

    for (size_t i = 0; i < local_elements; ++i)
    {
        if (std::fabs(C[i] - expected) > 0.001f)
        {
            local_correct = false;
            break;
        }
    }


    /*
     * ============================================================
     * GLOBAL VERIFICATION
     *
     * If any rank fails, global_status becomes 0.
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
     * The slowest MPI rank determines the total runtime.
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
     * PERFORMANCE CALCULATION
     * ============================================================
     */

    if (rank == 0)
    {
        /*
         * Matrix multiplication requires approximately:
         *
         *     2 * N^3
         *
         * floating-point operations.
         */

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
     * FREE SYCL MEMORY
     * ============================================================
     */

    sycl::free(d_A, q);
    sycl::free(d_B, q);
    sycl::free(d_C, q);


    /*
     * ============================================================
     * MPI FINALIZATION
     * ============================================================
     */

    MPI_Finalize();

    return 0;
}
