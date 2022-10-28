#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

double EXACT = 0.002747253;

int main(int argc, char **argv)
{
    int psize, prank, ierr, rand_seed;
    double result, eps;
    double x, y, z;


    eps = strtof(argv[1], NULL);
    rand_seed = strtol(argv[2], NULL, 10);
    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &psize);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    srand(prank * rand_seed);

    double res_time, time_elapsed = MPI_Wtime();

    double global_sum, local_sum = 0.0;
    long n = 1000 / (eps);
    long batch = n / psize;

    long i, it = 0;
    do
    {
        it++;
        for (i = 0; i < batch; i++)
        {
            x = (double) rand() / RAND_MAX;
            y = (double) rand() / RAND_MAX;
            z = (double) rand() / RAND_MAX;
            if ((x >= y) && (x * y >= z))
            {
                local_sum = local_sum + x * y * y * z * z * z;
            }
        }

        ierr = MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                             MPI_COMM_WORLD);

        result = global_sum / (batch * psize * it);

    }
    while(eps < fabs(EXACT - result));


    time_elapsed = MPI_Wtime() - time_elapsed;

    MPI_Reduce(&time_elapsed, &res_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                             MPI_COMM_WORLD);

    if (prank == 0)
    {
        double error = fabs(EXACT - result);
        printf("result:\t%.10lf\n", result);
        printf("error:\t%.10lf\n", error);
        printf("iters:\t%ld\n", it * n);
        printf("time:\t%.10lf\n", res_time);
    }

    ierr = MPI_Finalize();
    return 0;
}
