#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"
using namespace std;

#include <stdio.h>

// #################################################################

#define Mds(mi, i, j) As[(mi)*dim_bx * dim_by + (i)*dim_bx + j]
#define Nds(ni, i, j) As[TS * dim_bx * dim_by + (ni)*dim_bx * dim_by + (i)*dim_bx + j]
#define C(i, j) C[(i)*ldC + j]
#define A(i, j) A[(i)*ldA + j]
#define B(i, j) B[(i)*ldB + j]

// ################################################################

#define Mdss(mi, twy, twx, ty, tx) As[(mi * dim_by * TW_Y * TW_K) + ty * TW_Y * TW_K + twy * TW_K + tx]
#define Ndss(ni, twy, twx, ty, tx) As[(TS * dim_by * TW_Y * TW_K) + (ni * dim_bx * TW_X * TW_K) + ty * dim_bx * TW_X + tx * TW_X + twx]

// #################################################################
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

    int ldA = N, ldB = N, ldC = N; // A -> MxK, B -> KxN C-> MxN

    int row = by * dim_by + ty;
    int col = bx * dim_bx + tx;

    int npad_y = ceil(N / ((float)TW_Y * TS * dim_by)) * TW_Y * TS * dim_by;
    int npad_x = ceil(N / ((float)TW_X * TS * dim_bx)) * TW_X * TS * dim_bx;
    int npad_k = ceil(N / ((float)TW_K)) * TW_K;

    _FTYPE_ _c_[TS][TS][TW_Y][TW_X] = {0.0f};

    for (int tbase = 0; tbase < npad_k; tbase += TW_K)
    {

#pragma unroll
        for (int mi = 0; mi < TS; mi++)
        {
#pragma unroll
            for (int twy = 0; twy < TW_Y; twy++)
            {
                if ((row * TW_Y + twy + (mi * npad_y / TS)) < N && (tbase + tx) < N)
                    Mdss(mi, twy, twx, ty, tx) = A(row * TW_Y + twy + (mi * npad_y / TS), tbase + tx);
                else
                    Mdss(mi, twy, twx, ty, tx) = 0.0f;
            }
        }

#pragma unroll
        for (int ni = 0; ni < TS; ni++)
        {
#pragma unroll
            for (int twx = 0; twx < TW_X; twx++)
            {
                if ((tbase + ty) < N && (col * TW_X + twx + (ni * npad_x / TS)) < N)
                    Ndss(ni, twy, twx, ty, tx) = B(tbase + ty, col * TW_X + twx + (ni * npad_x / TS));
                else
                    Ndss(ni, twy, twx, ty, tx) = 0.0f;
            }
        }

        __syncthreads();

        for (int k = 0; k < TW_K; k++)
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
                            _c_[mi][ni][twy][twx] += Mdss(mi, twy, twx, ty, k) * Ndss(ni, twy, twx, k, tx);
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
                    if ((row * TW_Y + twy + (mi * npad_y / TS)) < N && (col * TW_X + twx + (ni * npad_x / TS)) < N)
                        C(row * TW_Y + twy + (mi * npad_y / TS), col * TW_X + twx + (ni * npad_x / TS)) = _c_[mi][ni][twy][twx];
                }
            }
        }
    }
}

__global__ void matmul_tilescale(int M, int K, int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B)
{

    extern __shared__ _FTYPE_ As[];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim_bx = blockDim.x;
    int dim_by = blockDim.y;

    const int TW = TILEDIM_K;
    const int TS = TILESCALE;

    int row = by * dim_by + ty;
    int col = bx * dim_bx + tx;
    int ldA = K, ldB = N, ldC = N; // A -> MxK, B -> KxN C-> MxN

    int mpad = ceil(M / ((float)TS)) * TS;
    int kpad = ceil(K / ((float)TW)) * TW;
    int npad = ceil(N / ((float)TS)) * TS;

    _FTYPE_ _c_[TS][TS] = {0.0f};

    for (int tbase = 0; tbase < kpad; tbase += TW)
    {

#pragma unroll
        for (int mi = 0; mi < TS; mi++)
        {
            if (row + mi * mpad / TS < M && tbase + tx < K)
                Mds(mi, ty, tx) = A(row + mi * mpad / TS, tbase + tx);
            else
                Mds(mi, ty, tx) = 0.0f;
        }

#pragma unroll
        for (int ni = 0; ni < TS; ni++)
        {
            if (tbase + ty < K && col + ni * npad / TS < N)
                Nds(ni, ty, tx) = B(tbase + ty, col + ni * npad / TS);
            else
                Nds(ni, ty, tx) = 0.0f;
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
                    _c_[mi][ni] += Mds(mi, ty, k) * Nds(ni, k, tx);
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
            if ((row + mi * mpad / TS) < M && (col + ni * npad / TS) < N)
                C(row + mi * mpad / TS, col + ni * npad / TS) = _c_[mi][ni];
        }
    }
}

__global__ void matmul_2dtile(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B)
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
    int ldA = N, ldB = N, ldC = N; // A -> MxK, B -> KxN C-> MxN

    int npad_y = ceil(N / ((float)TW_Y * dim_by)) * TW_Y * dim_by;
    int npad_x = ceil(N / ((float)TW_X * dim_bx)) * TW_X * dim_bx;
    int npad_k = ceil(N / ((float)TW_K)) * TW_K;

    _FTYPE_ _c_[TW_Y][TW_X] = {0.0f};

    for (int tbase = 0; tbase < npad_k; tbase += TW_K)
    {

#pragma unroll
        for (int twy = 0; twy < TW_Y; twy++)
        {
            if ((row * TW_Y + twy) < N && (tbase + tx) < N)
                Mdss(0, twy, twx, ty, tx) = A(row * TW_Y + twy, tbase + tx);
            else
                Mdss(0, twy, twx, ty, tx) = 0.0f;
        }

#pragma unroll

#pragma unroll
        for (int twx = 0; twx < TW_X; twx++)
        {
            if ((tbase + ty) < N && (col * TW_X + twx) < N)
                Ndss(0, twy, twx, ty, tx) = B(tbase + ty, col * TW_X + twx);
            else
                Ndss(0, twy, twx, ty, tx) = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TW_K; k++)
        {
#pragma unroll
            for (int twy = 0; twy < TW_Y; twy++)
            {
#pragma unroll
                for (int twx = 0; twx < TW_X; twx++)
                {
                    _c_[twy][twx] += Mdss(0, twy, twx, ty, k) * Ndss(0, twy, twx, k, tx);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int twy = 0; twy < TW_Y; twy++)
    {
#pragma unroll
        for (int twx = 0; twx < TW_X; twx++)
        {
            if ((row * TW_Y + twy) < N && (col * TW_X + twx) < N)
                C(row * TW_Y + twy, col * TW_X + twx) = _c_[twy][twx];
        }
    }
}
