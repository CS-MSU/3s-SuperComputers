#include <cmath>
#include <mpi.h>
#include "PuassonEquation.h"

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dims_size[2] = {0, 0};
    int periods[2] = {0, 0};
    int coords[2] = {0, 0};

    MPI_Comm comm;
    MPI_Dims_create(size, 2, dims_size);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims_size, periods,
                    1, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 2, coords);
    PuassonEquation::solve(
        atoi(argv[1]),
        atoi(argv[2]),
        dims_size[0],
        dims_size[1],
        0., 4., 0., 3., 1., 1., 0., 0.,
        comm);
    MPI_Finalize();
    return 0;
}
