#include "Matrix.h"

void Matrix::sub(Matrix &a, Matrix &b, Matrix &result)
{
    #pragma omp parallel for collapse(2)
    for (int i = a.yrange[0]; i < a.yrange[1]; ++i)
    {
        for (int j = a.xrange[0]; j < a.xrange[1]; ++j)
        {
            result(i, j, 0) = a(i, j, 0) - b(i, j, 0);
        }
    }
}

void Matrix::add(Matrix &a, Matrix &b, double alpha, double beta, Matrix &result)
{
    #pragma omp parallel for collapse(2)
    for (int i = a.yrange[0]; i < a.yrange[1]; ++i)
    {
        for (int j = a.xrange[0]; j < a.xrange[1]; ++j)
        {
            result(i, j, 0) = alpha * a(i, j, 0) + beta * b(i, j, 0);
        }
    }
}

void Matrix::copy(Matrix &a, Matrix &result)
{
    #pragma omp parallel for collapse(2)
    for (int i = a.yrange[0]; i < a.yrange[1]; ++i)
        for (int j = a.xrange[0]; j < a.xrange[1]; ++j)
            result(i, j, 0) = a(i, j, 0);
}

void Matrix::coef(Matrix &a, double alpha)
{
    #pragma omp parallel for collapse(2)
    for (int i = a.yrange[0]; i < a.yrange[1]; ++i)
        for (int j = a.xrange[0]; j < a.xrange[1]; ++j)
            a(i, j, 0) = alpha * a(i, j, 0);
}

double Matrix::getItem(int i, int j, int k) const
{
    i = i % y_size;
    j = j % x_size;
    return matrix[i * x_size * k_size + j * k_size + k];
}

double &Matrix::operator()(int i, int j, int k)
{
    i = i % y_size;
    j = j % x_size;
    return matrix[i * x_size * k_size + j * k_size + k];
}

void Matrix::mul(Matrix &A, Matrix &b, Matrix &top_vector, Matrix &bot_vector,
                 Matrix &left_vector, Matrix &right_vector, Matrix &result, int size_x, int size_y)
{
    int N = size_y;
    int M = size_x;
    double right_point, left_point, bot_point, top_point;
    const int start_y = b.yrange[0], end_y = b.yrange[1];
    const int start_x = b.xrange[0], end_x = b.xrange[1];
    #pragma omp parallel for collapse(2) private(right_point, left_point, bot_point, top_point)
    for (int i = start_y; i < end_y; ++i)
        for (int j = start_x; j < end_x; ++j)
        {
            // top points
            if (i == 0)
            {
                // left top corner
                if (j == 0)
                {
                    result(i, j, 0) =
                        A(i, j, 0) * b(i, j + 1, 0) +
                        A(i, j, 1) * b(i + 1, j, 0) +
                        A(i, j, 2) * b(i, j, 0);
                    continue;
                }
                // right top corner
                if (j == M - 1)
                {
                    result(i, j, 0) =
                        A(i, j, 0) * b(i, j - 1, 0) +
                        A(i, j, 1) * b(i + 1, j, 0) +
                        A(i, j, 2) * b(i, j, 0);
                    continue;
                }
                left_point = b(i, j - 1, 0);
                right_point = b(i, j + 1, 0);
                if (j == start_x)
                    left_point = left_vector(i, 0, 0);
                if (j == end_x - 1)
                    right_point = right_vector(i, 0, 0);
                result(i, j, 0) =
                    A(i, j, 0) * b(i + 1, j, 0) +
                    A(i, j, 1) * left_point +
                    A(i, j, 2) * right_point +
                    A(i, j, 3) * b(i, j, 0);
                continue;
            }
            // bot points
            if (i == N - 1)
            {
                // right bot
                if (j == M - 1)
                {
                    result(i, j, 0) =
                        A(i, j, 0) * b(i, j - 1, 0) +
                        A(i, j, 1) * b(i - 1, j, 0) +
                        A(i, j, 2) * b(i, j, 0);
                    continue;
                }
                // left bot
                if (j == 0)
                {
                    result(i, j, 0) =
                        A(i, j, 0) * b(i, j + 1, 0) +
                        A(i, j, 1) * b(i - 1, j, 0) +
                        A(i, j, 2) * b(i, j, 0);
                    continue;
                }
                // edge top
                left_point = b(i, j - 1, 0);
                right_point = b(i, j + 1, 0);
                if (j == start_x)
                    left_point = left_vector(i, 0, 0);
                if (j == end_x - 1)
                    right_point = right_vector(i, 0, 0);
                result(i, j, 0) =
                    A(i, j, 0) * b(i - 1, j, 0) +
                    A(i, j, 1) * left_point +
                    A(i, j, 2) * right_point +
                    A(i, j, 3) * b(i, j, 0);
                continue;
            }
            // left points
            if (j == 0)
            {
                // edge
                top_point = b(i - 1, j, 0);
                bot_point = b(i + 1, j, 0);
                if (i == start_y)
                    top_point = top_vector(0, j, 0);
                if (i == end_y - 1)
                    bot_point = bot_vector(0, j, 0);
                result(i, j, 0) =
                    A(i, j, 0) * b(i, j + 1, 0) +
                    A(i, j, 1) * bot_point +
                    A(i, j, 2) * top_point +
                    A(i, j, 3) * b(i, j, 0);
                continue;
            }
            // right points
            if (j == M - 1)
            {
                // edge
                bot_point = b(i + 1, j, 0);
                top_point = b(i - 1, j, 0);
                if (i == start_y)
                    top_point = top_vector(0, j, 0);
                if (i == end_y - 1)
                    bot_point = bot_vector(0, j, 0);
                result(i, j, 0) =
                    A(i, j, 0) * b(i, j - 1, 0) +
                    A(i, j, 1) * bot_point +
                    A(i, j, 2) * top_point +
                    A(i, j, 3) * b(i, j, 0);
                continue;
            }
            top_point = b(i - 1, j, 0);
            bot_point = b(i + 1, j, 0);
            left_point = b(i, j - 1, 0);
            right_point = b(i, j + 1, 0);
            if (j == start_x)
                left_point = left_vector(i, 0, 0);
            if (j == end_x - 1)
                right_point = right_vector(i, 0, 0);
            if (i == start_y)
                top_point = top_vector(0, j, 0);
            if (i == end_y - 1)
                bot_point = bot_vector(0, j, 0);
            // inner points
            result(i, j, 0) =
                A(i, j, 0) * left_point +
                A(i, j, 1) * right_point +
                A(i, j, 2) * top_point +
                A(i, j, 3) * bot_point +
                A(i, j, 4) * b(i, j, 0);
        }
}
