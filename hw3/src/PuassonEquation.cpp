#include "PuassonEquation.h"

double u_func(double x, double y)
{
    return sqrt(4 + x * y);
}

double k_func(double x, double y)
{
    return 4 + x + y;
}

double q_func(double x, double y)
{
    return x + y;
}

double f_func(double x, double y)
{
    return (
               (x + y) * (x * x + y * y + 4 * x * x * y * y + 30 * x * y + 56) + 4 * (x * x + y * y)) /
           (4 * (4 + x * y) * sqrt(4 + x * y));
}

double edge_top(double x, double y)
{
    return (x * x + 4 * x + x * y) / (2 * sqrt(4 + x * y));
}

double edge_bot(double x, double y)
{
    return -1 * (x * x + 4 * x + x * y) / (2 * sqrt(4 + x * y));
}

double edge_left(double x, double y)
{
    return (-1 * y * y - 4 * y + x * y + 8) / (2 * sqrt(4 + x * y));
}

double edge_right(double x, double y)
{
    return (y * y + 4 * y + 3 * x * y + 8) / (2 * sqrt(4 + x * y));
}

void PuassonEquation::set_true_solution(Matrix &a, double h1, double h2,
                                        double x_start, double y_start, int *xrange, int *yrange)
{
    double xi, yj;
    for (int i = yrange[0]; i < yrange[1]; ++i)
        for (int j = xrange[0]; j < xrange[1]; ++j)
        {
            xi = x_start + j * h1;
            yj = y_start + i * h2;
            a(i, j, 0) = u_func(xi, yj);
        }
}

void PuassonEquation::solve(int M, int N, int size_comm_x, int size_comm_y, double x_left, double x_right,
                            double y_bot, double y_top, double alpha_L, double alpha_R, double alpha_B, double alpha_T,
                            MPI_Comm comm)
{
    int xrange[2], yrange[2];
    int rank;
    int coords[2];
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 2, coords);

    xrange[0] = M / size_comm_x * coords[0];
    yrange[0] = N / size_comm_y * coords[1];
    xrange[1] = xrange[0] + M / size_comm_x;
    yrange[1] = yrange[0] + N / size_comm_y;

    Matrix matrix(M / size_comm_x,
                  N / size_comm_y, 5, comm);
    Matrix f_vector(M / size_comm_x,
                    N / size_comm_y, 1, comm);
    Matrix w_vector(M / size_comm_x,
                    N / size_comm_y, 1, comm);
    Matrix r_vector(M / size_comm_x,
                    N / size_comm_y, 1, comm);
    Matrix Ar_vector(M / size_comm_x,
                     N / size_comm_y, 1, comm);
    Matrix bot_vector(M / size_comm_x, 1, 1, comm);
    Matrix top_vector(M / size_comm_x, 1, 1, comm);
    Matrix left_vector(1, N / size_comm_y, 1, comm);
    Matrix right_vector(1, N / size_comm_y, 1, comm);
    Matrix true_solution(M / size_comm_x,
                         N / size_comm_y, 1, comm);
    double h1 = (double)(x_right - x_left) / (double)(M - 1);
    double h2 = (double)(y_top - y_bot) / (double)(N - 1);
    double x_start = x_left;
    double y_start = y_bot;

    set_true_solution(true_solution, h1, h2, x_start, y_start, xrange, yrange);

    MPI_Datatype row_type;
    MPI_Datatype column_type;

    MPI_Type_vector(1, N / size_comm_x, 1,
                    MPI_DOUBLE, &row_type);
    MPI_Type_vector(N / size_comm_y, 1, N / size_comm_x,
                    MPI_DOUBLE, &column_type);
    MPI_Type_commit(&row_type);
    MPI_Type_commit(&column_type);
    int rank_left = rank;
    int rank_right = rank;
    int rank_top = rank;
    int rank_bot = rank;
    int dims[2] = {coords[0], coords[1]};
    if (coords[0] != 0)
    {
        dims[0] -= 1;
        MPI_Cart_rank(comm, dims, &rank_left);
        dims[0] += 1;
    }
    if (coords[0] != size_comm_x - 1)
    {
        dims[0] += 1;
        MPI_Cart_rank(comm, dims, &rank_right);
        dims[0] -= 1;
    }
    if (coords[1] != 0)
    {
        dims[1] -= 1;
        MPI_Cart_rank(comm, dims, &rank_top);
        dims[1] += 1;
    }
    if (coords[1] != size_comm_y - 1)
    {
        dims[1] += 1;
        MPI_Cart_rank(comm, dims, &rank_bot);
        dims[1] -= 1;
    }

    double start_time = MPI_Wtime();
    filling(matrix, f_vector, N, M, h1, h2, x_start, y_start, xrange, yrange,
            alpha_L, alpha_R, alpha_B, alpha_T);
    optimize(matrix, f_vector, w_vector, r_vector, Ar_vector, true_solution,
             top_vector, bot_vector, left_vector, right_vector, N, M, h1, h2, xrange, yrange, coords, rank_left,
             rank_right, rank_bot, rank_top, rank, size_comm_x, size_comm_y, row_type, column_type, comm);
    double end_time = MPI_Wtime();
    double result_time, time = end_time - start_time;
    MPI_Reduce(&time, &result_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    Matrix::sub(true_solution, w_vector, r_vector);
    double error_e2_proc = 0., error_e2 = 0.;
    error_e2_proc = norm(r_vector, N, M, h1, h2, xrange, yrange);
    double error_max_proc, error_max;
    error_max_proc = norm_c(r_vector, xrange, yrange);
    MPI_Reduce(&error_e2_proc, &error_e2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&error_max_proc, &error_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        std::cout << "Time elapsed: " << result_time << std::endl;
        std::cout << "Absolute L2: " << sqrt(error_e2) << std::endl;
        std::cout << "Absolute Max: " << error_max << std::endl;
    }
}

void PuassonEquation::sendrecv(Matrix &b, Matrix &top_vector, Matrix &bot_vector, Matrix &left_vector, Matrix &right_vector,
                               int N, int M, int *coords, int rank_left, int rank_right, int rank_bot, int rank_top, int rank,
                               int size_comm_x, int size_comm_y, MPI_Datatype row_type, MPI_Datatype column_type,
                               MPI_Comm comm)
{
    int coord_x = coords[0], coord_y = coords[1];
    int x_left = 0, x_right = M / size_comm_x - 1;
    int y_bot = N / size_comm_y - 1, y_top = 0;
    MPI_Request request_right, request_left, request_bot, request_top;
    MPI_Status status_right, status_left, status_bot, status_top;
    if (coord_y == 0 or
        coord_y == size_comm_y - 1)
    {
        if (coord_y == 0)
        {
            // 0th row block
            MPI_Isend(&b(y_bot, 0, 0), 1, row_type,
                      rank_bot, rank, comm, &request_bot);
            MPI_Recv(&bot_vector(0, 0, 0), M / size_comm_x, MPI_DOUBLE,
                     rank_bot, rank_bot, comm, &status_bot);
        }
        // last row block
        if (coord_y == size_comm_y - 1)
        {
            MPI_Isend(&b(y_top, 0, 0), 1, row_type,
                      rank_top, rank, comm, &request_top);
            MPI_Recv(&top_vector(0, 0, 0), M / size_comm_x, MPI_DOUBLE,
                     rank_top, rank_top, comm, &status_top);
        }
    }
    else
    {
        // send row between first and last
        MPI_Isend(&b(y_bot, 0, 0), 1, row_type,
                  rank_bot, rank, comm, &request_bot);
        MPI_Isend(&b(y_top, 0, 0), 1, row_type,
                  rank_top, rank, comm, &request_top);
        // recv rows between first and last
        MPI_Recv(&top_vector(0, 0, 0), M / size_comm_x, MPI_DOUBLE,
                 rank_top, rank_top, comm, &status_top);
        MPI_Recv(&bot_vector(0, 0, 0), M / size_comm_x, MPI_DOUBLE,
                 rank_bot, rank_bot, comm, &status_bot);
    }

    // sendrecv vec to/from right cross x dimension
    if (coord_x == 0 or
        coord_x == size_comm_x - 1)
    {

        // 0th column block
        if (coord_x == 0)
        {
            MPI_Isend(&b(0, x_right, 0), 1, column_type,
                      rank_right, rank, comm, &request_right);
            MPI_Recv(&right_vector(0, 0, 0), N / size_comm_y, MPI_DOUBLE,
                     rank_right, rank_right, comm, &status_right);
        }
        // last column block
        if (coord_x == size_comm_x - 1)
        {
            MPI_Isend(&b(0, x_left, 0), 1, column_type,
                      rank_left, rank, comm, &request_left);
            MPI_Recv(&left_vector(0, 0, 0), N / size_comm_y, MPI_DOUBLE,
                     rank_left, rank_left, comm, &status_left);
        }
    }
    else
    {
        // send columns between first and last
        MPI_Isend(&b(0, x_right, 0), 1, column_type,
                  rank_right, rank, comm, &request_right);
        MPI_Isend(&b(0, x_left, 0), 1, column_type,
                  rank_left, rank, comm, &request_left);
        // recv columns between first and last
        MPI_Recv(&left_vector(0, 0, 0), N / size_comm_y, MPI_DOUBLE,
                 rank_left, rank_left, comm, &status_left);
        MPI_Recv(&right_vector(0, 0, 0), N / size_comm_y, MPI_DOUBLE,
                 rank_right, rank_right, comm, &status_right);
    }
}

double PuassonEquation::dot(Matrix &a, Matrix &b, int N, int M, double h1, double h2, int *xrange, int *yrange)
{
    double p1, p2;
    double sum_e2 = 0.0;
    // #pragma omp parallel for collapse(2) reduction(+:sum_e2)
    for (int i = yrange[0]; i < yrange[1]; ++i)
        for (int j = xrange[0]; j < xrange[1]; ++j)
        {
            if ((i == 0) or (i == N - 1))
                p2 = 0.5;
            else
                p2 = 1.;
            if ((j == 0) or (j == M - 1))
                p1 = 0.5;
            else
                p1 = 1.;
            sum_e2 += h1 * h2 * p1 * p2 * a(i, j, 0) * b(i, j, 0);
        }
    return sum_e2;
}

double PuassonEquation::norm_c(Matrix &a, int *xrange, int *yrange)
{
    double max_val = a(0, 0, 0);
    // #pragma omp parallel for collapse(2) reduction(max:max_val)
    for (int i = xrange[0]; i < xrange[1]; ++i)
        for (int j = yrange[0]; j < yrange[1]; ++j)
        {
            if (fabs(a(i, j, 0)) > max_val)
                max_val = fabs(a(i, j, 0));
        }
    return max_val;
}

void PuassonEquation::filling(Matrix &matrix, Matrix &f_vector, int N, int M, double h1, double h2,
                              double x_left, double y_bot, int *xrange, int *yrange,
                              double alpha_L, double alpha_R, double alpha_B, double alpha_T)
{
    double h1_2 = h1 * h1, h2_2 = h2 * h2;
    double xi, yj;
    for (int i = yrange[0]; i < yrange[1]; ++i)
        for (int j = xrange[0]; j < xrange[1]; ++j)
        {
            xi = x_left + j * h1;
            yj = y_bot + i * h2;
            // bottom points
            if (i == 0)
            {
                // left bottom
                if (j == 0)
                {
                    matrix(i, j, 0) =
                        -(2. / h1_2) * k_func(xi + 0.5 * h1, yj);
                    matrix(i, j, 1) =
                        -(2. / h2_2) * k_func(xi, yj + 0.5 * h2);
                    matrix(i, j, 2) =
                        -matrix(i, j, 0) -
                        matrix(i, j, 1) +
                        q_func(xi, yj) +
                        2. * alpha_L / h1 +
                        2. * alpha_B / h2;
                    f_vector(i, j, 0) =
                        f_func(xi, yj) +
                        (2. / h1 + 2. / h2) *
                            (h1 * edge_bot(xi, yj) +
                             h2 * edge_left(xi, yj)) /
                            (h1 + h2);
                    continue;
                }
                // right bottom
                if (j == M - 1)
                {
                    matrix(i, j, 0) =
                        -(2. / h1_2) * k_func(xi - 0.5 * h1, yj);
                    matrix(i, j, 1) =
                        -(2. / h2_2) * k_func(xi, yj + 0.5 * h2);
                    matrix(i, j, 2) =
                        -matrix(i, j, 0) -
                        matrix(i, j, 1) +
                        q_func(xi, yj) +
                        2. * alpha_R / h1 +
                        2. * alpha_B / h2;
                    f_vector(i, j, 0) =
                        f_func(xi, yj) +
                        (2. / h1 + 2. / h2) *
                            (h1 * edge_bot(xi, yj) +
                             h2 * edge_right(xi, yj)) /
                            (h1 + h2);
                    continue;
                }
                // edge
                matrix(i, j, 0) =
                    -(2. / h2_2) * k_func(xi, yj + 0.5 * h2);
                matrix(i, j, 1) =
                    -(1. / h1_2) * k_func(xi - 0.5 * h1, yj);
                matrix(i, j, 2) =
                    -(1. / h1_2) * k_func(xi + 0.5 * h1, yj);
                matrix(i, j, 3) =
                    -matrix(i, j, 0) -
                    matrix(i, j, 1) -
                    matrix(i, j, 2) +
                    q_func(xi, yj) +
                    2. * alpha_B / h2;
                f_vector(i, j, 0) =
                    f_func(xi, yj) + (2. / h2) * edge_bot(xi, yj);
                continue;
            }
            // top points
            if (i == N - 1)
            {
                // right top
                if (j == M - 1)
                {
                    matrix(i, j, 0) =
                        -(2. / h1_2) * k_func(xi - 0.5 * h1, yj);
                    matrix(i, j, 1) =
                        -(2. / h2_2) * k_func(xi, yj - 0.5 * h2);
                    matrix(i, j, 2) =
                        -matrix(i, j, 0) -
                        matrix(i, j, 1) +
                        q_func(xi, yj) +
                        2. * alpha_R / h1 +
                        2. * alpha_T / h2;
                    f_vector(i, j, 0) =
                        f_func(xi, yj) +
                        (2. / h1 + 2. / h2) *
                            (h1 * edge_top(xi, yj) +
                             h2 * edge_right(xi, yj)) /
                            (h1 + h2);
                    continue;
                }
                // left top
                if (j == 0)
                {
                    matrix(i, j, 0) =
                        -(2. / h1_2) * k_func(xi + 0.5 * h1, yj);
                    matrix(i, j, 1) =
                        -(2. / h2_2) * k_func(xi, yj - 0.5 * h2);
                    matrix(i, j, 2) =
                        -matrix(i, j, 0) -
                        matrix(i, j, 1) +
                        q_func(xi, yj) +
                        2. * alpha_L / h1 +
                        2. * alpha_T / h2;
                    f_vector(i, j, 0) =
                        f_func(xi, yj) +
                        (2. / h1 + 2. / h2) *
                            (h1 * edge_top(xi, yj) +
                             h2 * edge_left(xi, yj)) /
                            (h1 + h2);
                    continue;
                }
                // EDGE
                matrix(i, j, 0) =
                    -(2. / h2_2) * k_func(xi, yj - 0.5 * h2);
                matrix(i, j, 1) =
                    -(1. / h1_2) * k_func(xi - 0.5 * h1, yj);
                matrix(i, j, 2) =
                    -(1. / h1_2) * k_func(xi + 0.5 * h1, yj);
                matrix(i, j, 3) =
                    -matrix(i, j, 0) -
                    matrix(i, j, 1) -
                    matrix(i, j, 2) +
                    q_func(xi, yj) +
                    2. * alpha_T / h2;
                f_vector(i, j, 0) =
                    f_func(xi, yj) + (2. / h2) * edge_top(xi, yj);
                continue;
            }
            // left points
            if (j == 0)
            {
                // edge
                matrix(i, j, 0) =
                    -(2. / h1_2) * k_func(xi + 0.5 * h1, yj);
                matrix(i, j, 1) =
                    -(1. / h2_2) * k_func(xi, yj + 0.5 * h2);
                matrix(i, j, 2) =
                    -(1. / h2_2) * k_func(xi, yj - 0.5 * h2);
                matrix(i, j, 3) =
                    -matrix(i, j, 0) -
                    matrix(i, j, 1) -
                    matrix(i, j, 2) +
                    q_func(xi, yj) +
                    2. * alpha_L / h1;
                f_vector(i, j, 0) =
                    f_func(xi, yj) + (2. / h1) * edge_left(xi, yj);
                continue;
            }
            // right points
            if (j == M - 1)
            {
                // edge
                matrix(i, j, 0) =
                    -(2. / h1_2) * k_func(xi - 0.5 * h1, yj);
                matrix(i, j, 1) =
                    -(1. / h2_2) * k_func(xi, yj + 0.5 * h2);
                matrix(i, j, 2) =
                    -(1. / h2_2) * k_func(xi, yj - 0.5 * h2);
                matrix(i, j, 3) =
                    -matrix(i, j, 0) -
                    matrix(i, j, 1) -
                    matrix(i, j, 2) +
                    q_func(xi, yj) +
                    2. * alpha_R / h1;
                f_vector(i, j, 0) =
                    f_func(xi, yj) + (2. / h1) * edge_right(xi, yj);
                continue;
            }
            // inner points
            {
                matrix(i, j, 0) =
                    -1. / (h1_2)*k_func(xi - 0.5 * h1, yj);
                matrix(i, j, 1) =
                    -1. / (h1_2)*k_func(xi + 0.5 * h1, yj);
                matrix(i, j, 2) =
                    -1. / (h2_2)*k_func(xi, yj - 0.5 * h2);
                matrix(i, j, 3) =
                    -1. / (h2_2)*k_func(xi, yj + 0.5 * h2);
                matrix(i, j, 4) =
                    -matrix(i, j, 0) - matrix(i, j, 1) -
                    matrix(i, j, 2) - matrix(i, j, 3) +
                    q_func(xi, yj);
                f_vector(i, j, 0) = f_func(xi, yj);
            }
        }
}

void PuassonEquation::optimize(Matrix &matrix, Matrix &f_vector, Matrix &w_vector,
                               Matrix &r_vector, Matrix &Ar_vector, Matrix &true_solution,
                               Matrix &top_vector, Matrix &bot_vector, Matrix &left_vector,
                               Matrix &right_vector, int N, int M, double h1, double h2,
                               int *xrange, int *yrange, int *coords, int rank_left,
                               int rank_right, int rank_bot, int rank_top, int rank,
                               int size_comm_x, int size_comm_y,
                               MPI_Datatype row_type, MPI_Datatype column_type, MPI_Comm comm)
{

    double tau_ar_proc[2] = {0., 0.}, tau_ar[2] = {0., 0.}, tau_k;
    double eps_proc = 0., eps = 10.;
    int iter = 0;
    while (eps > 1e-6)
    {
        PuassonEquation::sendrecv(w_vector, top_vector, bot_vector, left_vector, right_vector,
                                  N, M, coords, rank_left, rank_right, rank_bot, rank_top, rank,
                                  size_comm_x, size_comm_y, row_type, column_type, comm);
        Matrix::mul(matrix, w_vector, top_vector, bot_vector,
                    left_vector, right_vector, Ar_vector, M, N);
        Matrix::sub(Ar_vector, f_vector, r_vector);
        PuassonEquation::sendrecv(r_vector, top_vector, bot_vector, left_vector, right_vector,
                                  N, M, coords, rank_left, rank_right, rank_bot, rank_top, rank,
                                  size_comm_x, size_comm_y, row_type, column_type, comm);
        Matrix::mul(matrix, r_vector, top_vector, bot_vector,
                    left_vector, right_vector, Ar_vector, M, N);
        tau_ar_proc[0] = dot(Ar_vector, r_vector, N, M, h1, h2, xrange, yrange);
        tau_ar_proc[1] = dot(Ar_vector, Ar_vector, N, M, h1, h2, xrange, yrange);
        MPI_Allreduce(&tau_ar_proc, &tau_ar, 2, MPI_DOUBLE, MPI_SUM, comm);
        tau_k = tau_ar[0] / tau_ar[1];
        Matrix::add(w_vector, r_vector, 1.0, -tau_k, w_vector);
        // e2 norm
        eps_proc = norm(r_vector, N, M, h1, h2, xrange, yrange);
        MPI_Allreduce(&eps_proc, &eps, 1, MPI_DOUBLE, MPI_SUM, comm);
        eps = sqrt(eps);
        ++iter;
	if (iter > 10000) break;
    }
    if (rank == 0)
    {
        std::cout << "iter: " << iter << std::endl;
    }
}
