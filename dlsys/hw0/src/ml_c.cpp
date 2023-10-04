#include <cmath>
#include <iostream>
using namespace std;

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
    float *B = new float[batch_size * n];

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
            Z[j * k + (y_batch[j] - '0')] -= 1.0;
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

int main()
{

    float *A = new float[6]{1, 2, 3, 4, 5, 6};
    display(A, 2, 3);

    const unsigned char *y = new unsigned char[2]{'1', '2'};

    display(y, 2, 1);

    auto tA = transpose(A, 2, 3);
    display(tA, 3, 2);

    cout << "--------X-------" << endl;

    float *X = new float[6]{1, 2, 3, 4, 5, 6};
    display(X, 2, 3);

    cout << "--------Y-------" << endl;

    float *Y = new float[6]{1, 2, 3, 4, 5, 6};
    display(Y, 3, 2);

    cout << "---------------RESULT----------------" << endl;

    float *Z = mat_mul(X, Y, 2, 3, 2);
    display(Z, 2, 2);

    float *E = new float[6]{.1, .1, .3, .9, .5, .8};

    display(normexp(E, 3, 2), 3, 2);

    // be careful to track the dimensions
    // auto B = transpose(A, 2, 3);
    // display(B, 3, 2);

    return 0;
}