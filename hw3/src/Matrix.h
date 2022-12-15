#ifndef PARALLEL_MATRIX_H
#define PARALLEL_MATRIX_H
#include "mpi.h"
#include <omp.h>

class Matrix
{
public:
    int x_size, y_size, k_size;
    int xrange[2];
    int yrange[2];
    int coords[2];
    int rank;
    int size;
    double *matrix;

    Matrix() = default;

    Matrix(Matrix const &) = default;

    Matrix(int x_size, int y_size, int k_size, MPI_Comm comm)
    {
        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 2, coords);
        this->x_size = x_size;
        this->y_size = y_size;
        this->k_size = k_size;
        this->size = x_size * y_size * k_size;
        xrange[0] = x_size * coords[0];
        yrange[0] = y_size * coords[1];
        xrange[1] = xrange[0] + x_size;
        yrange[1] = yrange[0] + y_size;
        this->matrix = new double[size];
        for (int i = 0; i < size; ++i)
            this->matrix[i] = 0.0;
    }

    int getSizeX() const
    {
        return this->x_size;
    }

    int getSizeY() const
    {
        return this->y_size;
    }

    int getSizeK() const
    {
        return this->k_size;
    }

    static void sub(Matrix &a, Matrix &b, Matrix &result);

    static void add(Matrix &a, Matrix &b, double alpha, double beta, Matrix &result);

    static void copy(Matrix &a, Matrix &result);

    static void mul(Matrix &A, Matrix &b, Matrix &top_vector, Matrix &bot_vector,
                    Matrix &left_vector, Matrix &right_vector, Matrix &result, int size_x, int size_y);

    static void coef(Matrix &A, double alpha);

    double getItem(int i, int j, int k) const;

    double &operator()(int i, int j, int k);
};

#endif // PARALLEL_MATRIX_H
