#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"
using namespace std;

#include <stdio.h>

// #################################################################

__global__ void matMul_naive(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B)
{

    int I = blockIdx.y * blockDim.y + threadIdx.y;
    int J = blockIdx.x * blockDim.x + threadIdx.x;

    if ((I < N) && (J < N))
    {
        _FTYPE_ _c = 0;
        for (unsigned int k = 0; k < N; k++)
        {
            _FTYPE_ a = A[I * N + k];
            _FTYPE_ b = B[k * N + J];
            _c += a * b;
        }
        C[I * N + J] = _c;
    }
}

__global__ void matMul_1x1(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B)
{
    extern __shared__ _FTYPE_ As[];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim_bx = blockDim.x;
    int dim_by = blockDim.y;

    int I = by * dim_by + ty; // Row
    int J = bx * dim_bx + tx; // Col

    int TW = TILEDIM_K;
    int TS = TILESCALE;
    int npad = ceil(N / ((float)TW * TS)) * TW * TS;

    _FTYPE_ _c = 0;
    for (int tbase = 0; tbase < npad; tbase += TW)
    {

        if ((I < N) && (tbase + tx) < N)
            As[ty * dim_by + tx] = A[I * N + tbase + tx];
        else
            As[ty * dim_by + tx] = 0.0f;

        if ((tbase + ty) < N && (J < N))
            As[dim_bx * dim_by + ty * dim_by + tx] = B[(tbase + ty) * N + J];
        else
            As[dim_bx * dim_by + ty * dim_by + tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TW; k++)
        {
            _c += As[ty * dim_by + k] * As[dim_bx * dim_by + k * dim_by + tx];
        }
        __syncthreads();

        if ((I < N) && (J < N))
            C[I * N + J] = _c;
    }
}

__global__ void matMul_2x2(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B)
{

    extern __shared__ _FTYPE_ As[];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim_bx = blockDim.x;
    int dim_by = blockDim.y;

    int row = by * dim_by + ty;
    int col = bx * dim_bx + tx;

    int TW = TILEDIM_K;
    int TS = TILESCALE;
    int npad = ceil(N / ((float)TW * TS)) * TW * TS;

    _FTYPE_ _c_n0 = 0;
    _FTYPE_ _c_n1 = 0;
    _FTYPE_ _c_m0 = 0;
    _FTYPE_ _c_m1 = 0;

    if ((row < (npad / TS)) && (col < (npad / TS)))
    {

        for (int tbase = 0; tbase < npad; tbase += TW)
        {
            if (row < N && tbase + tx < N)
                As[ty * dim_by + tx] = A[row * N + tbase + tx];
            else
                As[ty * dim_by + tx] = 0.0;

            if (row + npad / 2 < N && (tbase + tx) < N)
                As[dim_bx * dim_by + ty * dim_by + tx] = A[((row + npad / 2) * N + tbase + tx)];
            else
                As[dim_bx * dim_by + ty * dim_by + tx] = 0.0;

            if (tbase + ty < N && col < N)
                As[2 * dim_bx * dim_by + ty * dim_by + tx] = B[(tbase + ty) * N + col];
            else
                As[2 * dim_bx * dim_by + ty * dim_by + tx] = 0.0;

            if ((tbase + ty) < N && (col + npad / 2 < N))
                As[3 * dim_bx * dim_by + ty * dim_by + tx] = B[(tbase + ty) * N + col + (npad / 2)];
            else
                As[3 * dim_bx * dim_by + ty * dim_by + tx] = 0.0f;

            __syncthreads();

            for (int k = 0; k < TW; k++)
            {
                _c_n0 += As[ty * dim_by + k] * As[2 * dim_bx * dim_by + k * dim_by + tx];
                _c_n1 += As[ty * dim_by + k] * As[3 * dim_bx * dim_by + k * dim_by + tx];
                _c_m0 += As[dim_bx * dim_by + ty * dim_by + k] * As[2 * dim_bx * dim_by + k * dim_by + tx];
                _c_m1 += As[dim_bx * dim_by + ty * dim_by + k] * As[3 * dim_bx * dim_by + k * dim_by + tx];
            }
            __syncthreads();
        }
        C[row * N + col] = _c_n0;
        if ((row < N) && (col + (npad / TS)) < N)
            C[row * N + (col + (npad / TS))] = _c_n1;
        if (((row + (npad / TS)) < N) && (col < N))
            C[(row + (npad / TS)) * N + col] = _c_m0;
        if (((row + (npad / TS)) < N) && ((col + (npad / TS)) < N))
            C[(row + (npad / TS)) * N + (col + (npad / TS))] = _c_m1;
    }
}

__global__ void matMul_2x2_v2(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B)
{

    extern __shared__ _FTYPE_ As[];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim_bx = blockDim.x;
    int dim_by = blockDim.y;

    int row = by * dim_by + ty;
    int col = bx * dim_bx + tx;

    const int TW = TILEDIM_K;
    const int TS = TILESCALE;

    int npad = ceil(N / ((float)TW * TS)) * TW * TS;

    _FTYPE_ _c_[TS][TS] = {0};

    if (row < npad / TS && col < npad / TS)
    {

        for (int tbase = 0; tbase < npad; tbase += TW)
        {

#pragma unroll
            for (int mi = 0; mi < TS; mi++)
            {
                if (row + mi * npad / TS < N && tbase + tx < N)
                    As[mi * dim_bx * dim_by + ty * dim_by + tx] = A[(row + (mi * npad / TS)) * N + tbase + tx];
                else
                    As[mi * dim_bx * dim_by + ty * dim_by + tx] = 0.0;
            }

#pragma unroll
            for (int ni = 0; ni < TS; ni++)
            {
                if (tbase + ty < N && col + ni * npad / TS < N)
                    As[(TS + ni) * dim_bx * dim_by + ty * dim_by + tx] = B[(tbase + ty) * N + col + (ni * npad / TS)];
                else
                    As[(TS + ni) * dim_bx * dim_by + ty * dim_by + tx] = 0.0;
            }
            __syncthreads();

            for (int k = 0; k < TW; k++)
            {
#pragma unroll
                for (int mi = 0; mi < TS; mi++)
                {
#pragma unroll
                    for (int ni = 0; ni < TS; ni++)
                    {
                        _c_[mi][ni] += As[mi * dim_bx * dim_by + ty * dim_by + k] * As[(TS + ni) * dim_bx * dim_by + k * dim_by + tx];
                    }
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (int mi = 0; mi < TS; mi++)
        {
#pragma unroll
            for (int ni = 0; ni < TS; ni++)
            {
                if ((row + mi * npad / TS) < N && (col + ni * npad / TS) < N)
                    C[(row + mi * npad / TS) * N + (col + ni * npad / TS)] = _c_[mi][ni];
            }
        }
    }
}

__global__ void matMul_8x8(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B)
{

    extern __shared__ _FTYPE_ As[];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim_bx = blockDim.x;
    int dim_by = blockDim.y;

    int I = by * dim_by + ty;
    int J = bx * dim_bx + tx;
    int TW = TILEDIM_K;

    if ((I < (N / 8)) && (J < (N / 8)))
    {
        _FTYPE_ _c_m0n0 = 0;
        _FTYPE_ _c_m0n1 = 0;
        _FTYPE_ _c_m0n2 = 0;
        _FTYPE_ _c_m0n3 = 0;
        _FTYPE_ _c_m0n4 = 0;
        _FTYPE_ _c_m0n5 = 0;
        _FTYPE_ _c_m0n6 = 0;
        _FTYPE_ _c_m0n7 = 0;

        _FTYPE_ _c_m1n0 = 0;
        _FTYPE_ _c_m1n1 = 0;
        _FTYPE_ _c_m1n2 = 0;
        _FTYPE_ _c_m1n3 = 0;
        _FTYPE_ _c_m1n4 = 0;
        _FTYPE_ _c_m1n5 = 0;
        _FTYPE_ _c_m1n6 = 0;
        _FTYPE_ _c_m1n7 = 0;

        _FTYPE_ _c_m2n0 = 0;
        _FTYPE_ _c_m2n1 = 0;
        _FTYPE_ _c_m2n2 = 0;
        _FTYPE_ _c_m2n3 = 0;
        _FTYPE_ _c_m2n4 = 0;
        _FTYPE_ _c_m2n5 = 0;
        _FTYPE_ _c_m2n6 = 0;
        _FTYPE_ _c_m2n7 = 0;

        _FTYPE_ _c_m3n0 = 0;
        _FTYPE_ _c_m3n1 = 0;
        _FTYPE_ _c_m3n2 = 0;
        _FTYPE_ _c_m3n3 = 0;
        _FTYPE_ _c_m3n4 = 0;
        _FTYPE_ _c_m3n5 = 0;
        _FTYPE_ _c_m3n6 = 0;
        _FTYPE_ _c_m3n7 = 0;

        _FTYPE_ _c_m4n0 = 0;
        _FTYPE_ _c_m4n1 = 0;
        _FTYPE_ _c_m4n2 = 0;
        _FTYPE_ _c_m4n3 = 0;
        _FTYPE_ _c_m4n4 = 0;
        _FTYPE_ _c_m4n5 = 0;
        _FTYPE_ _c_m4n6 = 0;
        _FTYPE_ _c_m4n7 = 0;

        _FTYPE_ _c_m5n0 = 0;
        _FTYPE_ _c_m5n1 = 0;
        _FTYPE_ _c_m5n2 = 0;
        _FTYPE_ _c_m5n3 = 0;
        _FTYPE_ _c_m5n4 = 0;
        _FTYPE_ _c_m5n5 = 0;
        _FTYPE_ _c_m5n6 = 0;
        _FTYPE_ _c_m5n7 = 0;

        _FTYPE_ _c_m6n0 = 0;
        _FTYPE_ _c_m6n1 = 0;
        _FTYPE_ _c_m6n2 = 0;
        _FTYPE_ _c_m6n3 = 0;
        _FTYPE_ _c_m6n4 = 0;
        _FTYPE_ _c_m6n5 = 0;
        _FTYPE_ _c_m6n6 = 0;
        _FTYPE_ _c_m6n7 = 0;

        _FTYPE_ _c_m7n0 = 0;
        _FTYPE_ _c_m7n1 = 0;
        _FTYPE_ _c_m7n2 = 0;
        _FTYPE_ _c_m7n3 = 0;
        _FTYPE_ _c_m7n4 = 0;
        _FTYPE_ _c_m7n5 = 0;
        _FTYPE_ _c_m7n6 = 0;
        _FTYPE_ _c_m7n7 = 0;

        for (unsigned int tile = 0; tile < N / TW; tile++)
        {
            As[ty * dim_by + tx] = A[I * N + tile * TW + tx];
            As[dim_bx * dim_by + ty * dim_by + tx] = A[(I + (N / 8)) * N + tile * TW + tx];
            As[2 * dim_bx * dim_by + ty * dim_by + tx] = A[(I + (N / 4)) * N + tile * TW + tx];
            As[3 * dim_bx * dim_by + ty * dim_by + tx] = A[(I + (3 * N / 8)) * N + tile * TW + tx];
            As[4 * dim_bx * dim_by + ty * dim_by + tx] = A[(I + (N / 2)) * N + tile * TW + tx];
            As[5 * dim_bx * dim_by + ty * dim_by + tx] = A[(I + (5 * N / 8)) * N + tile * TW + tx];
            As[6 * dim_bx * dim_by + ty * dim_by + tx] = A[(I + (3 * N / 4)) * N + tile * TW + tx];
            As[7 * dim_bx * dim_by + ty * dim_by + tx] = A[(I + (7 * N / 8)) * N + tile * TW + tx];

            As[8 * dim_bx * dim_by + ty * dim_by + tx] = B[(tile * TW + ty) * N + J];
            As[9 * dim_bx * dim_by + ty * dim_by + tx] = B[(tile * TW + ty) * N + J + (N / 8)];
            As[10 * dim_bx * dim_by + ty * dim_by + tx] = B[(tile * TW + ty) * N + J + (N / 4)];
            As[11 * dim_bx * dim_by + ty * dim_by + tx] = B[(tile * TW + ty) * N + J + (3 * N / 8)];
            As[12 * dim_bx * dim_by + ty * dim_by + tx] = B[(tile * TW + ty) * N + J + (N / 2)];
            As[13 * dim_bx * dim_by + ty * dim_by + tx] = B[(tile * TW + ty) * N + J + (5 * N / 8)];
            As[14 * dim_bx * dim_by + ty * dim_by + tx] = B[(tile * TW + ty) * N + J + (3 * N / 4)];
            As[15 * dim_bx * dim_by + ty * dim_by + tx] = B[(tile * TW + ty) * N + J + (7 * N / 8)];
            __syncthreads();

            for (unsigned int k = 0; k < TW; k++)
            {
                _c_m0n0 += As[ty * dim_by + k] * As[8 * dim_bx * dim_by + k * dim_by + tx];
                _c_m0n1 += As[ty * dim_by + k] * As[9 * dim_bx * dim_by + k * dim_by + tx];
                _c_m0n2 += As[ty * dim_by + k] * As[10 * dim_bx * dim_by + k * dim_by + tx];
                _c_m0n3 += As[ty * dim_by + k] * As[11 * dim_bx * dim_by + k * dim_by + tx];
                _c_m0n4 += As[ty * dim_by + k] * As[12 * dim_bx * dim_by + k * dim_by + tx];
                _c_m0n5 += As[ty * dim_by + k] * As[13 * dim_bx * dim_by + k * dim_by + tx];
                _c_m0n6 += As[ty * dim_by + k] * As[14 * dim_bx * dim_by + k * dim_by + tx];
                _c_m0n7 += As[ty * dim_by + k] * As[15 * dim_bx * dim_by + k * dim_by + tx];

                _c_m1n0 += As[dim_bx * dim_by + ty * dim_by + k] * As[8 * dim_bx * dim_by + k * dim_by + tx];
                _c_m1n1 += As[dim_bx * dim_by + ty * dim_by + k] * As[9 * dim_bx * dim_by + k * dim_by + tx];
                _c_m1n2 += As[dim_bx * dim_by + ty * dim_by + k] * As[10 * dim_bx * dim_by + k * dim_by + tx];
                _c_m1n3 += As[dim_bx * dim_by + ty * dim_by + k] * As[11 * dim_bx * dim_by + k * dim_by + tx];
                _c_m1n4 += As[dim_bx * dim_by + ty * dim_by + k] * As[12 * dim_bx * dim_by + k * dim_by + tx];
                _c_m1n5 += As[dim_bx * dim_by + ty * dim_by + k] * As[13 * dim_bx * dim_by + k * dim_by + tx];
                _c_m1n6 += As[dim_bx * dim_by + ty * dim_by + k] * As[14 * dim_bx * dim_by + k * dim_by + tx];
                _c_m1n7 += As[dim_bx * dim_by + ty * dim_by + k] * As[15 * dim_bx * dim_by + k * dim_by + tx];

                _c_m2n0 += As[2 * dim_bx * dim_by + ty * dim_by + k] * As[8 * dim_bx * dim_by + k * dim_by + tx];
                _c_m2n1 += As[2 * dim_bx * dim_by + ty * dim_by + k] * As[9 * dim_bx * dim_by + k * dim_by + tx];
                _c_m2n2 += As[2 * dim_bx * dim_by + ty * dim_by + k] * As[10 * dim_bx * dim_by + k * dim_by + tx];
                _c_m2n3 += As[2 * dim_bx * dim_by + ty * dim_by + k] * As[11 * dim_bx * dim_by + k * dim_by + tx];
                _c_m2n4 += As[2 * dim_bx * dim_by + ty * dim_by + k] * As[12 * dim_bx * dim_by + k * dim_by + tx];
                _c_m2n5 += As[2 * dim_bx * dim_by + ty * dim_by + k] * As[13 * dim_bx * dim_by + k * dim_by + tx];
                _c_m2n6 += As[2 * dim_bx * dim_by + ty * dim_by + k] * As[14 * dim_bx * dim_by + k * dim_by + tx];
                _c_m2n7 += As[2 * dim_bx * dim_by + ty * dim_by + k] * As[15 * dim_bx * dim_by + k * dim_by + tx];

                _c_m3n0 += As[3 * dim_bx * dim_by + ty * dim_by + k] * As[8 * dim_bx * dim_by + k * dim_by + tx];
                _c_m3n1 += As[3 * dim_bx * dim_by + ty * dim_by + k] * As[9 * dim_bx * dim_by + k * dim_by + tx];
                _c_m3n2 += As[3 * dim_bx * dim_by + ty * dim_by + k] * As[10 * dim_bx * dim_by + k * dim_by + tx];
                _c_m3n3 += As[3 * dim_bx * dim_by + ty * dim_by + k] * As[11 * dim_bx * dim_by + k * dim_by + tx];
                _c_m3n4 += As[3 * dim_bx * dim_by + ty * dim_by + k] * As[12 * dim_bx * dim_by + k * dim_by + tx];
                _c_m3n5 += As[3 * dim_bx * dim_by + ty * dim_by + k] * As[13 * dim_bx * dim_by + k * dim_by + tx];
                _c_m3n6 += As[3 * dim_bx * dim_by + ty * dim_by + k] * As[14 * dim_bx * dim_by + k * dim_by + tx];
                _c_m3n7 += As[3 * dim_bx * dim_by + ty * dim_by + k] * As[15 * dim_bx * dim_by + k * dim_by + tx];

                _c_m4n0 += As[4 * dim_bx * dim_by + ty * dim_by + k] * As[8 * dim_bx * dim_by + k * dim_by + tx];
                _c_m4n1 += As[4 * dim_bx * dim_by + ty * dim_by + k] * As[9 * dim_bx * dim_by + k * dim_by + tx];
                _c_m4n2 += As[4 * dim_bx * dim_by + ty * dim_by + k] * As[10 * dim_bx * dim_by + k * dim_by + tx];
                _c_m4n3 += As[4 * dim_bx * dim_by + ty * dim_by + k] * As[11 * dim_bx * dim_by + k * dim_by + tx];
                _c_m4n4 += As[4 * dim_bx * dim_by + ty * dim_by + k] * As[12 * dim_bx * dim_by + k * dim_by + tx];
                _c_m4n5 += As[4 * dim_bx * dim_by + ty * dim_by + k] * As[13 * dim_bx * dim_by + k * dim_by + tx];
                _c_m4n6 += As[4 * dim_bx * dim_by + ty * dim_by + k] * As[14 * dim_bx * dim_by + k * dim_by + tx];
                _c_m4n7 += As[4 * dim_bx * dim_by + ty * dim_by + k] * As[15 * dim_bx * dim_by + k * dim_by + tx];

                _c_m5n0 += As[5 * dim_bx * dim_by + ty * dim_by + k] * As[8 * dim_bx * dim_by + k * dim_by + tx];
                _c_m5n1 += As[5 * dim_bx * dim_by + ty * dim_by + k] * As[9 * dim_bx * dim_by + k * dim_by + tx];
                _c_m5n2 += As[5 * dim_bx * dim_by + ty * dim_by + k] * As[10 * dim_bx * dim_by + k * dim_by + tx];
                _c_m5n3 += As[5 * dim_bx * dim_by + ty * dim_by + k] * As[11 * dim_bx * dim_by + k * dim_by + tx];
                _c_m5n4 += As[5 * dim_bx * dim_by + ty * dim_by + k] * As[12 * dim_bx * dim_by + k * dim_by + tx];
                _c_m5n5 += As[5 * dim_bx * dim_by + ty * dim_by + k] * As[13 * dim_bx * dim_by + k * dim_by + tx];
                _c_m5n6 += As[5 * dim_bx * dim_by + ty * dim_by + k] * As[14 * dim_bx * dim_by + k * dim_by + tx];
                _c_m5n7 += As[5 * dim_bx * dim_by + ty * dim_by + k] * As[15 * dim_bx * dim_by + k * dim_by + tx];

                _c_m6n0 += As[6 * dim_bx * dim_by + ty * dim_by + k] * As[8 * dim_bx * dim_by + k * dim_by + tx];
                _c_m6n1 += As[6 * dim_bx * dim_by + ty * dim_by + k] * As[9 * dim_bx * dim_by + k * dim_by + tx];
                _c_m6n2 += As[6 * dim_bx * dim_by + ty * dim_by + k] * As[10 * dim_bx * dim_by + k * dim_by + tx];
                _c_m6n3 += As[6 * dim_bx * dim_by + ty * dim_by + k] * As[11 * dim_bx * dim_by + k * dim_by + tx];
                _c_m6n4 += As[6 * dim_bx * dim_by + ty * dim_by + k] * As[12 * dim_bx * dim_by + k * dim_by + tx];
                _c_m6n5 += As[6 * dim_bx * dim_by + ty * dim_by + k] * As[13 * dim_bx * dim_by + k * dim_by + tx];
                _c_m6n6 += As[6 * dim_bx * dim_by + ty * dim_by + k] * As[14 * dim_bx * dim_by + k * dim_by + tx];
                _c_m6n7 += As[6 * dim_bx * dim_by + ty * dim_by + k] * As[15 * dim_bx * dim_by + k * dim_by + tx];

                _c_m7n0 += As[7 * dim_bx * dim_by + ty * dim_by + k] * As[8 * dim_bx * dim_by + k * dim_by + tx];
                _c_m7n1 += As[7 * dim_bx * dim_by + ty * dim_by + k] * As[9 * dim_bx * dim_by + k * dim_by + tx];
                _c_m7n2 += As[7 * dim_bx * dim_by + ty * dim_by + k] * As[10 * dim_bx * dim_by + k * dim_by + tx];
                _c_m7n3 += As[7 * dim_bx * dim_by + ty * dim_by + k] * As[11 * dim_bx * dim_by + k * dim_by + tx];
                _c_m7n4 += As[7 * dim_bx * dim_by + ty * dim_by + k] * As[12 * dim_bx * dim_by + k * dim_by + tx];
                _c_m7n5 += As[7 * dim_bx * dim_by + ty * dim_by + k] * As[13 * dim_bx * dim_by + k * dim_by + tx];
                _c_m7n6 += As[7 * dim_bx * dim_by + ty * dim_by + k] * As[14 * dim_bx * dim_by + k * dim_by + tx];
                _c_m7n7 += As[7 * dim_bx * dim_by + ty * dim_by + k] * As[15 * dim_bx * dim_by + k * dim_by + tx];
            }
            __syncthreads();
        }
        C[I * N + J] = _c_m0n0;
        C[I * N + (J + (N / 8))] = _c_m0n1;
        C[I * N + (J + (N / 4))] = _c_m0n2;
        C[I * N + (J + (3 * N / 8))] = _c_m0n3;
        C[I * N + (J + (N / 2))] = _c_m0n4;
        C[I * N + (J + (5 * N / 8))] = _c_m0n5;
        C[I * N + (J + (3 * N / 4))] = _c_m0n6;
        C[I * N + (J + (7 * N / 8))] = _c_m0n7;

        C[(I + (N / 8)) * N + J] = _c_m1n0;
        C[(I + (N / 8)) * N + (J + (N / 8))] = _c_m1n1;
        C[(I + (N / 8)) * N + (J + (N / 4))] = _c_m1n2;
        C[(I + (N / 8)) * N + (J + (3 * N / 8))] = _c_m1n3;
        C[(I + (N / 8)) * N + (J + (N / 2))] = _c_m1n4;
        C[(I + (N / 8)) * N + (J + (5 * N / 8))] = _c_m1n5;
        C[(I + (N / 8)) * N + (J + (3 * N / 4))] = _c_m1n6;
        C[(I + (N / 8)) * N + (J + (7 * N / 8))] = _c_m1n7;

        C[(I + (N / 4)) * N + J] = _c_m2n0;
        C[(I + (N / 4)) * N + (J + (N / 8))] = _c_m2n1;
        C[(I + (N / 4)) * N + (J + (N / 4))] = _c_m2n2;
        C[(I + (N / 4)) * N + (J + (3 * N / 8))] = _c_m2n3;
        C[(I + (N / 4)) * N + (J + (N / 2))] = _c_m2n4;
        C[(I + (N / 4)) * N + (J + (5 * N / 8))] = _c_m2n5;
        C[(I + (N / 4)) * N + (J + (3 * N / 4))] = _c_m2n6;
        C[(I + (N / 4)) * N + (J + (7 * N / 8))] = _c_m2n7;

        C[(I + (3 * N / 8)) * N + J] = _c_m3n0;
        C[(I + (3 * N / 8)) * N + (J + (N / 8))] = _c_m3n1;
        C[(I + (3 * N / 8)) * N + (J + (N / 4))] = _c_m3n2;
        C[(I + (3 * N / 8)) * N + (J + (3 * N / 8))] = _c_m3n3;
        C[(I + (3 * N / 8)) * N + (J + (N / 2))] = _c_m3n4;
        C[(I + (3 * N / 8)) * N + (J + (5 * N / 8))] = _c_m3n5;
        C[(I + (3 * N / 8)) * N + (J + (3 * N / 4))] = _c_m3n6;
        C[(I + (3 * N / 8)) * N + (J + (7 * N / 8))] = _c_m3n7;

        C[(I + (N / 2)) * N + J] = _c_m4n0;
        C[(I + (N / 2)) * N + (J + (N / 8))] = _c_m4n1;
        C[(I + (N / 2)) * N + (J + (N / 4))] = _c_m4n2;
        C[(I + (N / 2)) * N + (J + (3 * N / 8))] = _c_m4n3;
        C[(I + (N / 2)) * N + (J + (N / 2))] = _c_m4n4;
        C[(I + (N / 2)) * N + (J + (5 * N / 8))] = _c_m4n5;
        C[(I + (N / 2)) * N + (J + (3 * N / 4))] = _c_m4n6;
        C[(I + (N / 2)) * N + (J + (7 * N / 8))] = _c_m4n7;

        C[(I + (5 * N / 8)) * N + J] = _c_m5n0;
        C[(I + (5 * N / 8)) * N + (J + (N / 8))] = _c_m5n1;
        C[(I + (5 * N / 8)) * N + (J + (N / 4))] = _c_m5n2;
        C[(I + (5 * N / 8)) * N + (J + (3 * N / 8))] = _c_m5n3;
        C[(I + (5 * N / 8)) * N + (J + (N / 2))] = _c_m5n4;
        C[(I + (5 * N / 8)) * N + (J + (5 * N / 8))] = _c_m5n5;
        C[(I + (5 * N / 8)) * N + (J + (3 * N / 4))] = _c_m5n6;
        C[(I + (5 * N / 8)) * N + (J + (7 * N / 8))] = _c_m5n7;

        C[(I + (3 * N / 4)) * N + J] = _c_m6n0;
        C[(I + (3 * N / 4)) * N + (J + (N / 8))] = _c_m6n1;
        C[(I + (3 * N / 4)) * N + (J + (N / 4))] = _c_m6n2;
        C[(I + (3 * N / 4)) * N + (J + (3 * N / 8))] = _c_m6n3;
        C[(I + (3 * N / 4)) * N + (J + (N / 2))] = _c_m6n4;
        C[(I + (3 * N / 4)) * N + (J + (5 * N / 8))] = _c_m6n5;
        C[(I + (3 * N / 4)) * N + (J + (3 * N / 4))] = _c_m6n6;
        C[(I + (3 * N / 4)) * N + (J + (7 * N / 8))] = _c_m6n7;

        C[(I + (7 * N / 8)) * N + J] = _c_m7n0;
        C[(I + (7 * N / 8)) * N + (J + (N / 8))] = _c_m7n1;
        C[(I + (7 * N / 8)) * N + (J + (N / 4))] = _c_m7n2;
        C[(I + (7 * N / 8)) * N + (J + (3 * N / 8))] = _c_m7n3;
        C[(I + (7 * N / 8)) * N + (J + (N / 2))] = _c_m7n4;
        C[(I + (7 * N / 8)) * N + (J + (5 * N / 8))] = _c_m7n5;
        C[(I + (7 * N / 8)) * N + (J + (3 * N / 4))] = _c_m7n6;
        C[(I + (7 * N / 8)) * N + (J + (7 * N / 8))] = _c_m7n7;
    }
}

__global__ void matmul_tilescale(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B)
{

    extern __shared__ _FTYPE_ As[];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim_bx = blockDim.x;
    int dim_by = blockDim.y;

    int row = by * dim_by + ty;
    int col = bx * dim_bx + tx;

    const int TW = TILEDIM_K;
    const int TS = TILESCALE;

    int npad = ceil(N / ((float)TW * TS)) * TW * TS;

    _FTYPE_ _c_0[TS][TS] = {0};
    _FTYPE_ _c_1[TS][TS] = {0};

    if (row < npad / TS && col < npad / TS)
    {

        for (int tbase = 0; tbase < npad; tbase += TW)
        {

#pragma unroll
            for (int mi = 0; mi < TS; mi++)
            {
                if (row + mi * npad / TS < N && tbase + tx < N)
                    As[mi * dim_bx * dim_by + ty * dim_bx + tx] = A[(row + (mi * npad / TS)) * N + tbase + tx];
                else
                    As[mi * dim_bx * dim_by + ty * dim_bx + tx] = 0.0;
            }

#pragma unroll
            for (int ni = 0; ni < TS; ni++)
            {
                if (tbase + ty < N && col + ni * npad / TS < N)
                    As[(TS + ni) * dim_bx * dim_by + ty * dim_bx + tx] = B[(tbase + ty) * N + col + (ni * npad / TS)];
                else
                    As[(TS + ni) * dim_bx * dim_by + ty * dim_bx + tx] = 0.0;
            }
            __syncthreads();

            for (int k = 0; k < TW; k++)
            {
#pragma unroll
                for (int mi = 0; mi < TS; mi++)
                {
#pragma unroll
                    for (int ni = 0; ni < TS; ni++)
                    {
                        _c_0[mi][ni] += As[mi * dim_bx * dim_by + ty * dim_bx + k] * As[(TS + ni) * dim_bx * dim_by + k * dim_bx + tx];
                        // printf("bx %d by %d tx %d ty %d row %d col %d _c[%d,%d] %f As[%d] %f As[%d] %f\n", bx, by, tx, ty, row, col,mi,ni,_c_0[mi][ni],mi * dim_bx * dim_by + ty * dim_bx + k,As[mi * dim_bx * dim_by + ty * dim_bx + k],(TS + ni) * dim_bx * dim_by + k * dim_bx + tx,As[(TS + ni) * dim_bx * dim_by + k * dim_bx + tx]);
                    }
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (int mi = 0; mi < TS; mi++)
        {
#pragma unroll
            for (int ni = 0; ni < TS; ni++)
            {
                if ((row + mi * npad / TS) < N && (col + ni * npad / TS) < N)
                    C[(row + mi * npad / TS) * N + (col + ni * npad / TS)] = _c_0[mi][ni];
                // printf("bx %d by %d tx %d ty %d row %d col %d C[%d,%d] %f _c[%d,%d] %f \n", bx, by, tx, ty, row, col, row + mi * npad / TS, col + ni * npad / TS,C[(row + mi * npad / TS) * N + (col + ni * npad / TS)],mi,ni,_c_0[mi][ni]);
            }
        }
    }
}

__global__ void matmul_2dtile_tilescale(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B)
{

    extern __shared__ _FTYPE_ As[];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim_bx = blockDim.x;
    int dim_by = blockDim.y;

    const int TW_K = TILEDIM_K;
    const int TW_X = TILEDIM_X;
    const int TW_Y = TILEDIM_Y;
    const int TS = TILESCALE;

    int row = by * dim_by + ty;
    int col = bx * dim_bx + tx;

    int npad_y = ceil(N / ((float)TW_Y * TS * dim_by)) * TW_Y * TS * dim_by;
    int npad_x = ceil(N / ((float)TW_X * TS * dim_bx)) * TW_X * TS * dim_bx;
    int npad_k = ceil(N / ((float)TW_K)) * TW_K;

    _FTYPE_ _c_[TS][TS][TW_X][TW_Y] = {0};

    if ((row < npad_y / TS) && (col < npad_x / TS))
    {

        for (int tbase = 0; tbase < npad_k; tbase += TW_K)
        {

#pragma unroll
            for (int mi = 0; mi < TS; mi++)
            {
#pragma unroll
                for (int twy = 0; twy < TW_Y; twy++)
                {
#pragma unroll
                    for (int mtwk = 0; mtwk < TW_K; mtwk++)
                    {
                        // if ((row*TW_Y + twy + (mi * npad_y / TS)) < N && (tbase + mtwk) < N)
                        As[(mi * dim_by * TW_Y * TW_K) + (ty * TW_Y * TW_K) + (twy * TW_K) + mtwk] = A[(row * TW_Y + twy + (mi * npad_y / TS)) * N + tbase + mtwk];
                        // else
                        //     As[(mi * dim_by * TW_Y * TW_K) + (ty * TW_Y * TW_K) + (twy * TW_K) + mtwk] = 0.0;

                        // printf(" SM Fetch A n %d %d %d %d bx %d by %d tx %d ty %d row %d col %d mi %d twy %d mtwk %d As[%d] %f A[%d,%d] %f \n",N,npad_x,npad_y,npad_k,bx,by,tx,ty,row,col,mi,twy,mtwk,(mi * dim_by * TW_Y * TW_K) + (ty * TW_Y * TW_K) + (twy * TW_K) + mtwk,As[(mi * dim_by * TW_Y * TW_K) + (ty * TW_Y * TW_K) + (twy * TW_K) + mtwk],(row*TW_Y + twy + (mi * npad_y / TS)),(tbase + mtwk),A[(row*TW_Y + twy + (mi * npad_y / TS)) * N + tbase + mtwk]);
                    }
                }
            }

#pragma unroll
            for (int ni = 0; ni < TS; ni++)
            {
#pragma unroll
                for (int ntwk = 0; ntwk < TW_K; ntwk++)
                {
#pragma unroll
                    for (int twx = 0; twx < TW_X; twx++)
                    {
                        // if ((tbase + ntwk) < N && (col*TW_X + twx + (ni * npad_x / TS)) < N)
                        As[(TS * dim_by * TW_Y * TW_K) + (ni * dim_bx * TW_X * TW_K) + (ntwk * dim_bx * TW_X) + (tx * TW_X) + twx] = B[(tbase + ntwk) * N + col * TW_X + twx + (ni * npad_x / TS)];
                        // else
                        //     As[(TS * dim_by * TW_Y * TW_K) + (ni * dim_bx * TW_X * TW_K) + (ntwk * dim_bx * TW_X) + (tx * TW_X) + twx ] = 0.0;
                        // printf(" SM Fetch B n %d %d %d %d bx %d by %d tx %d ty %d row %d col %d ni %d twx %d ntwk %d As[%d] %f B[%d,%d] %f \n",N,npad_x,npad_y,npad_k,bx,by,tx,ty,row,col,ni,twx,ntwk,(TS * dim_by * TW_Y * TW_K) + (ni * dim_bx * TW_X * TW_K) + (ntwk * dim_bx * TW_X) + (tx * TW_X) + twx,As[(TS * dim_by * TW_Y * TW_K) + (ni * dim_bx * TW_X * TW_K) + (ntwk * dim_bx * TW_X) + (tx * TW_X) + twx],(tbase + ntwk),(col*TW_X + twx + (ni * npad_x / TS)),B[(tbase + ntwk) * N + col*TW_X + twx + (ni * npad_x / TS)]);
                    }
                }
            }

            __syncthreads();

            for (int twk = 0; twk < TW_K; twk++)
            {
#pragma unroll
                for (int mi = 0; mi < TS; mi++)
                {
#pragma unroll
                    for (int ni = 0; ni < TS; ni++)
                    {
#pragma unroll
                        for (int twy = 0; twy < TW_Y; twy++)
                        {
#pragma unroll
                            for (int twx = 0; twx < TW_X; twx++)
                            {
                                _c_[mi][ni][twy][twx] += As[(mi * dim_by * TW_Y * TW_K) + (ty * TW_Y * TW_K) + (twy * TW_K) + twk] * As[(TS * dim_by * TW_Y * TW_K) + (ni * dim_bx * TW_X * TW_K) + ((tx * TW_X) + twx) + (TW_X * dim_bx) * twk];
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (int mi = 0; mi < TS; mi++)
        {
#pragma unroll
            for (int ni = 0; ni < TS; ni++)
            {
#pragma unroll
                for (int twy = 0; twy < TW_Y; twy++)
                {
#pragma unroll
                    for (int twx = 0; twx < TW_X; twx++)
                    {
                        // if ((row*TW_Y + twy + (mi * npad_y / TS)) < N && (col*TW_X + twx + (ni * npad_x / TS)) < N)
                        C[(row * TW_Y + twy + (mi * npad_y / TS)) * N + (col * TW_X + twx + (ni * npad_x / TS))] = _c_[mi][ni][twy][twx];
                        // printf("\n C store bx %d by %d tx %d ty %d row %d col %d C[%d,%d] %f _c[%d][%d][%d][%d] %f \n",bx,by,tx,ty,row,col,(row*TW_Y + twy + (mi * npad_y / TS)),(col*TW_X + twx + (ni * npad_x / TS)),C[(row*TW_Y + twy + (mi * npad_y / TS)) * N + (col*TW_X + twx + (ni * npad_x / TS))],mi,ni,twy,twx,_c_[mi][ni][twy][twx]);
                    }
                }
            }
        }
    }
}
