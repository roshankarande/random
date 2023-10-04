#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

float *mat_mul(const float *A, const float *B, int m, int n, int p)
{

    int i = 0, j = 0, k = 0;

    float *C = new float[m * p]{0};

    for (i = 0; i < m; i++)
    {
        for (j = 0; j < p; j++)
        {
            for (k = 0; k < n; k++)
            {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }

    return C;
}

void display(const float *A, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << A[i * n + j] << " ";
        }
        std::cout << "\n";
    }
}

void display(const unsigned char *A, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << A[i * n + j] - '0' << " ";
        }
        std::cout << "\n";
    }
}

float *transpose(const float *A, int m, int n)
{
    float *AT = new float[m * n]{0};
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            AT[m * j + i] = A[n * i + j];
        }
    }

    return AT;
}

float *normexp(const float *X, int m, int n)
{
    float *result = new float[m * n]{0};
    float *rowSum = new float[m]{0};

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            result[n * i + j] = exp(X[n * i + j]);
            rowSum[i] += result[n * i + j];
        }
    }

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            result[n * i + j] /= rowSum[i];
        }
    }

    return result;
}

float *get_batch(const float *X, int start, int n, int batch_size)
{
    float *B = new float[n * batch_size];

    for (int i = 0; i < batch_size; i++)
    {
        for (int j = 0; j < n; j++)
        {
            B[i * n + j] = X[(start + i) * n + j];
        }
    }

    return B;
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, int m, int n, int k,
                                  float lr, int batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row   
     *          major (C) format
     *     m (int): number of examples
     *     n (int): input dimension
     *     k (int): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns: (None)
     *
     **/

    int total_batches = (int)(m / batch);

    for (int i = 0; i < total_batches; i++)
    {
        auto X_batch = get_batch(X, i * batch, n, batch);
        char *y_batch = new char[batch];
        for (int j = 0; j < batch; j++)
        {
            y_batch[j] = y[i * batch + j];
        }

        auto Z = normexp(mat_mul(X_batch, theta, batch, n, k), batch, k);
        float *X_batch_T = transpose(X_batch, batch, n);

        for (int j = 0; j < batch; j++)
        {
            Z[j * k + y_batch[j]] -= 1.0;
        }

        float *mat_result = mat_mul(X_batch_T, Z, n, batch, k);

        for (int s = 0; s < n; s++)
        {
            for (int t = 0; t < k; t++)
            {
                theta[s * k + t] -= lr * mat_result[s * k + t] / batch;
            }
        }

        // delete[] Z;
        delete[] mat_result;
        delete[] X_batch;
        delete[] y_batch;
        delete[] X_batch_T;
    }
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m)
{
    m.def(
        "softmax_regression_epoch_cpp",
        [](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch)
        {
            softmax_regression_epoch_cpp(
                static_cast<const float *>(X.request().ptr),
                static_cast<const unsigned char *>(y.request().ptr),
                static_cast<float *>(theta.request().ptr),
                X.request().shape[0],
                X.request().shape[1],
                theta.request().shape[1],
                lr,
                batch);
        },
        py::arg("X"), py::arg("y"), py::arg("theta"),
        py::arg("lr"), py::arg("batch"));
}
