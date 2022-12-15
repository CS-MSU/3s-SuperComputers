#ifndef PARALLEL_PUASSONEQUATION_H
#define PARALLEL_PUASSONEQUATION_H

#include "Matrix.h"
#include <omp.h>
#include <cmath>
#include <iostream>

class PuassonEquation
{

    static void sendrecv(Matrix &b, Matrix &top_vector, Matrix &bot_vector, Matrix &left_vector, Matrix &right_vector,
                         int N, int M, int *coords, int rank_left, int rank_right, int rank_bot, int rank_top, int rank,
                         int size_comm_x, int size_comm_y, MPI_Datatype row_type, MPI_Datatype column_type,
                         MPI_Comm comm);

    static double dot(Matrix &a, Matrix &b, int N, int M, double h1, double h2, int *xrange, int *yrange);

    static double norm_c(Matrix &a, int *xrange, int *yrange);

    static double norm(Matrix &a, int N, int M, double h1, double h2, int *xrange, int *yrange)
    {
        return dot(a, a, N, M, h1, h2, xrange, yrange);
    }

    static void filling(Matrix &matrix, Matrix &f_vector, int N, int M, double h1, double h2,
                        double x_left, double y_bot, int *xrange, int *yrange,
                        double alpha_L, double alpha_R, double alpha_B, double alpha_T);

    static void optimize(Matrix &matrix, Matrix &f_vector, Matrix &w_vector,
                         Matrix &r_vector, Matrix &Ar_vector, Matrix &true_solution,
                         Matrix &top_vector, Matrix &bot_vector, Matrix &left_vector,
                         Matrix &right_vector, int N, int M, double h1, double h2,
                         int *xrange, int *yrange, int *coords, int rank_left,
                         int rank_right, int rank_bot, int rank_top, int rank,
                         int size_comm_x, int size_comm_y,
                         MPI_Datatype row_type, MPI_Datatype column_type, MPI_Comm comm);

    static void set_true_solution(Matrix &a, double h1, double h2,
                                  double x_start, double y_start, int *xrange, int *yrange);

public:
    static void solve(int M, int N, int size_comm_x, int size_comm_y, double x_left, double x_right,
                      double y_bot, double y_top, double alpha_L, double alpha_R, double alpha_B, double alpha_T,
                      MPI_Comm comm);
};

#endif // PARALLEL_PUASSONEQUATION_H
