#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"
using namespace std;

#include <stdio.h>

#ifdef NAIVE
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B)
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

#else
// You should be changing the kernel here for the non naive implementation.

__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B)
{

    extern __shared__ _FTYPE_ As[];
    // extern __shared__ _FTYPE_ Bs[];

    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    int block_dim_x = blockDim.x;
    int block_dim_y = blockDim.y;

    int I = block_y * block_dim_y + thread_y;
    int J = block_x * block_dim_x + thread_x;
    int npad = ceil(N / ((float)TILEDIM_K * 8)) * TILEDIM_K * 8;

    // printf("block_x %d thread_x %d block_y %d thread_y %d I %d J %d \n",block_x,thread_x,block_y,thread_y,I,J);
    // printf("Thread Dim x %d y %d Block Dim %d %d \n",block_dim_x,block_dim_y,block_dim_x,block_dim_y);
    if ((I < (npad / 8)) && (J < (npad / 8)))
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

        for (unsigned int c_tile = 0; c_tile < N / TILEDIM_K; c_tile++)
        {
            // if(){
            if ((I < N) && ((c_tile * TILEDIM_K + thread_x) < N))
                As[thread_y * block_dim_y + thread_x] = A[I * N + c_tile * TILEDIM_K + thread_x];
            else
                As[thread_y * block_dim_y + thread_x] = 0.0;
            if (((I + (N / 8)) < N) && ((c_tile * TILEDIM_K + thread_x) < N))
                As[block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = A[(I + (N / 8)) * N + c_tile * TILEDIM_K + thread_x];
            else
                As[block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = 0.0;
            if (((I + (N / 4)) < N) && ((c_tile * TILEDIM_K + thread_x) < N))
                As[2 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = A[(I + (N / 4)) * N + c_tile * TILEDIM_K + thread_x];
            else
                As[2 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = 0.0;
            if (((I + (3 * N / 8)) < N) && ((c_tile * TILEDIM_K + thread_x) < N))
                As[3 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = A[(I + (3 * N / 8)) * N + c_tile * TILEDIM_K + thread_x];
            else
                As[3 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = 0.0;
            if (((I + (N / 2)) < N) && ((c_tile * TILEDIM_K + thread_x) < N))
                As[4 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = A[(I + (N / 2)) * N + c_tile * TILEDIM_K + thread_x];
            else
                As[4 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = 0.0;
            if (((I + (5 * N / 8)) < N) && ((c_tile * TILEDIM_K + thread_x) < N))
                As[5 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = A[(I + (5 * N / 8)) * N + c_tile * TILEDIM_K + thread_x];
            else
                As[5 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = 0.0;
            if (((I + (3 * N / 4)) < N) && ((c_tile * TILEDIM_K + thread_x) < N))
                As[6 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = A[(I + (3 * N / 4)) * N + c_tile * TILEDIM_K + thread_x];
            else
                As[6 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = 0.0;
            if (((I + (7 * N / 8)) < N) && ((c_tile * TILEDIM_K + thread_x) < N))
                As[7 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = A[(I + (7 * N / 8)) * N + c_tile * TILEDIM_K + thread_x];
            else
                As[7 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = 0.0;
            //}

            if (((c_tile * TILEDIM_K + thread_y) < N) && ((J) < N))
                As[8 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = B[(c_tile * TILEDIM_K + thread_y) * N + J];
            else
                As[8 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = 0.0;
            if (((c_tile * TILEDIM_K + thread_y) < N) && ((J + (N / 8)) < N))
                As[9 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = B[(c_tile * TILEDIM_K + thread_y) * N + J + (N / 8)];
            else
                As[9 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = 0.0;
            if (((c_tile * TILEDIM_K + thread_y) < N) && ((J + (N / 4)) < N))
                As[10 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = B[(c_tile * TILEDIM_K + thread_y) * N + J + (N / 4)];
            else
                As[10 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = 0.0;
            if (((c_tile * TILEDIM_K + thread_y) < N) && ((J + (3 * N / 8)) < N))
                As[11 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = B[(c_tile * TILEDIM_K + thread_y) * N + J + (3 * N / 8)];
            else
                As[11 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = 0.0;
            if (((c_tile * TILEDIM_K + thread_y) < N) && ((J + (N / 2)) < N))
                As[12 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = B[(c_tile * TILEDIM_K + thread_y) * N + J + (N / 2)];
            else
                As[12 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = 0.0;
            if (((c_tile * TILEDIM_K + thread_y) < N) && ((J + (5 * N / 8)) < N))
                As[13 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = B[(c_tile * TILEDIM_K + thread_y) * N + J + (5 * N / 8)];
            else
                As[13 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = 0.0;
            if (((c_tile * TILEDIM_K + thread_y) < N) && ((J + (3 * N / 4)) < N))
                As[14 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = B[(c_tile * TILEDIM_K + thread_y) * N + J + (3 * N / 4)];
            else
                As[14 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = 0.0;
            if (((c_tile * TILEDIM_K + thread_y) < N) && ((J + (7 * N / 8)) < N))
                As[15 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = B[(c_tile * TILEDIM_K + thread_y) * N + J + (7 * N / 8)];
            else
                As[15 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = 0.0;
            __syncthreads();

            // printf("before block_x %d block_y %d thread_x %d thread_y %d I %d J %d c_tile %d _c_m0n0 %f _c_m0n1 %f As[%d] %f A[%d] %f Bs[%d] %f B[%d] %f Bs_1[%d] %f B[%d] %f\n",block_x,block_y,thread_x,thread_y,I,J,c_tile, _c_m0n0,_c_m0n1,thread_y*block_dim_y+thread_x,As[thread_y*block_dim_y+thread_x],I*N + c_tile*TILEDIM_K + thread_x,A[I*N + c_tile*TILEDIM_K + thread_x],block_dim_x*block_dim_y+thread_y*block_dim_y+thread_x, As[block_dim_x*block_dim_y+thread_y*block_dim_y+thread_x],(c_tile*TILEDIM_K + thread_y)*N + J,B[(c_tile*TILEDIM_K + thread_y)*N + J],2*block_dim_x*block_dim_y + thread_y*block_dim_y+thread_x,As[2*block_dim_x*block_dim_y + thread_y*block_dim_y+thread_x],(c_tile*TILEDIM_K + thread_y)*N + J + 2,B[(c_tile*TILEDIM_K + thread_y)*N + J + (N/2)]);

            for (unsigned int k = 0; k < TILEDIM_K; k++)
            {
                _c_m0n0 += As[thread_y * block_dim_y + k] * As[8 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m0n1 += As[thread_y * block_dim_y + k] * As[9 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m0n2 += As[thread_y * block_dim_y + k] * As[10 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m0n3 += As[thread_y * block_dim_y + k] * As[11 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m0n4 += As[thread_y * block_dim_y + k] * As[12 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m0n5 += As[thread_y * block_dim_y + k] * As[13 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m0n6 += As[thread_y * block_dim_y + k] * As[14 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m0n7 += As[thread_y * block_dim_y + k] * As[15 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];

                _c_m1n0 += As[block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[8 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m1n1 += As[block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[9 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m1n2 += As[block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[10 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m1n3 += As[block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[11 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m1n4 += As[block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[12 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m1n5 += As[block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[13 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m1n6 += As[block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[14 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m1n7 += As[block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[15 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];

                _c_m2n0 += As[2 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[8 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m2n1 += As[2 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[9 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m2n2 += As[2 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[10 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m2n3 += As[2 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[11 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m2n4 += As[2 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[12 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m2n5 += As[2 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[13 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m2n6 += As[2 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[14 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m2n7 += As[2 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[15 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];

                _c_m3n0 += As[3 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[8 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m3n1 += As[3 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[9 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m3n2 += As[3 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[10 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m3n3 += As[3 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[11 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m3n4 += As[3 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[12 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m3n5 += As[3 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[13 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m3n6 += As[3 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[14 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m3n7 += As[3 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[15 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];

                _c_m4n0 += As[4 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[8 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m4n1 += As[4 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[9 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m4n2 += As[4 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[10 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m4n3 += As[4 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[11 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m4n4 += As[4 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[12 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m4n5 += As[4 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[13 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m4n6 += As[4 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[14 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m4n7 += As[4 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[15 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];

                _c_m5n0 += As[5 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[8 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m5n1 += As[5 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[9 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m5n2 += As[5 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[10 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m5n3 += As[5 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[11 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m5n4 += As[5 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[12 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m5n5 += As[5 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[13 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m5n6 += As[5 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[14 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m5n7 += As[5 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[15 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];

                _c_m6n0 += As[6 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[8 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m6n1 += As[6 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[9 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m6n2 += As[6 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[10 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m6n3 += As[6 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[11 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m6n4 += As[6 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[12 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m6n5 += As[6 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[13 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m6n6 += As[6 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[14 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m6n7 += As[6 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[15 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];

                _c_m7n0 += As[7 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[8 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m7n1 += As[7 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[9 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m7n2 += As[7 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[10 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m7n3 += As[7 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[11 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m7n4 += As[7 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[12 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m7n5 += As[7 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[13 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m7n6 += As[7 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[14 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m7n7 += As[7 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[15 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];

                // printf("block_x %d block_y %d thread_x %d thread_y %d I %d J %d c_tile %d k %d A[%d] %f B[%d] %f As[%d] %f Bs_0[%d] %f Bs_1[%d] %f\n",block_x,block_y,thread_x,thread_y,I,J,c_tile,k,I*N + c_tile*TILEDIM_K + thread_x,A[I*N + c_tile*TILEDIM_K + thread_x],(c_tile*TILEDIM_K + thread_y)*N + J,B[(c_tile*TILEDIM_K + thread_y)*N + J],thread_y*block_dim_y+k,As[thread_y*block_dim_y+k],block_dim_x*block_dim_y + k*block_dim_y+thread_x,As[(block_dim_x*block_dim_y) + k*block_dim_y+thread_x],block_dim_x*block_dim_y + k*block_dim_y+block_dim_x*block_dim_y+thread_x,As[block_dim_x*block_dim_y + k*block_dim_y+block_dim_x*block_dim_y+thread_x]);
            }
            __syncthreads();
            // printf("block_x %d block_y %d thread_x %d thread_y %d I %d J %d c_tile %d _c_m0n0 %f _c_m0n1 %f \n",block_x,block_y,thread_x,thread_y,I,J,c_tile, _c_m0n0,_c_m0n1);
        }
        C[I * N + J] = _c_m0n0;
        if ((J + (N / 8)) < N)
            C[I * N + (J + (N / 8))] = _c_m0n1;
        if ((J + (N / 4)) < N)
            C[I * N + (J + (N / 4))] = _c_m0n2;
        if ((J + (3 * N / 8)) < N)
            C[I * N + (J + (3 * N / 8))] = _c_m0n3;
        if ((J + (N / 2)) < N)
            C[I * N + (J + (N / 2))] = _c_m0n4;
        if ((J + (5 * N / 8)) < N)
            C[I * N + (J + (5 * N / 8))] = _c_m0n5;
        if ((J + (3 * N / 4)) < N)
            C[I * N + (J + (3 * N / 4))] = _c_m0n6;
        if ((J + (7 * N / 8)) < N)
            C[I * N + (J + (7 * N / 8))] = _c_m0n7;

        if (((I + (N / 8)) < N) && (J < N))
            C[(I + (N / 8)) * N + J] = _c_m1n0;
        if (((I + (N / 8)) < N) && ((J + (N / 8)) < N))
            C[(I + (N / 8)) * N + (J + (N / 8))] = _c_m1n1;
        if (((I + (N / 8)) < N) && ((J + (N / 4)) < N))
            C[(I + (N / 8)) * N + (J + (N / 4))] = _c_m1n2;
        if (((I + (N / 8)) < N) && ((J + (3 * N / 8)) < N))
            C[(I + (N / 8)) * N + (J + (3 * N / 8))] = _c_m1n3;
        if (((I + (N / 8)) < N) && ((J + (N / 2)) < N))
            C[(I + (N / 8)) * N + (J + (N / 2))] = _c_m1n4;
        if (((I + (N / 8)) < N) && ((J + (5 * N / 8)) < N))
            C[(I + (N / 8)) * N + (J + (5 * N / 8))] = _c_m1n5;
        if (((I + (N / 8)) < N) && ((J + (3 * N / 4)) < N))
            C[(I + (N / 8)) * N + (J + (3 * N / 4))] = _c_m1n6;
        if (((I + (N / 8)) < N) && ((J + (7 * N / 8)) < N))
            C[(I + (N / 8)) * N + (J + (7 * N / 8))] = _c_m1n7;

        if (((I + (N / 4)) < N) && (J < N))
            C[(I + (N / 4)) * N + J] = _c_m2n0;
        if (((I + (N / 4)) < N) && ((J + (N / 8)) < N))
            C[(I + (N / 4)) * N + (J + (N / 8))] = _c_m2n1;
        if (((I + (N / 4)) < N) && ((J + (N / 4)) < N))
            C[(I + (N / 4)) * N + (J + (N / 4))] = _c_m2n2;
        if (((I + (N / 4)) < N) && ((J + (3 * N / 8)) < N))
            C[(I + (N / 4)) * N + (J + (3 * N / 8))] = _c_m2n3;
        if (((I + (N / 4)) < N) && ((J + (N / 2)) < N))
            C[(I + (N / 4)) * N + (J + (N / 2))] = _c_m2n4;
        if (((I + (N / 4)) < N) && ((J + (5 * N / 8)) < N))
            C[(I + (N / 4)) * N + (J + (5 * N / 8))] = _c_m2n5;
        if (((I + (N / 4)) < N) && ((J + (3 * N / 4)) < N))
            C[(I + (N / 4)) * N + (J + (3 * N / 4))] = _c_m2n6;
        if (((I + (N / 4)) < N) && ((J + (7 * N / 8)) < N))
            C[(I + (N / 4)) * N + (J + (7 * N / 8))] = _c_m2n7;

        if (((I + (3 * N / 8)) < N) && (J < N))
            C[(I + (3 * N / 8)) * N + J] = _c_m3n0;
        if (((I + (3 * N / 8)) < N) && ((J + (N / 8)) < N))
            C[(I + (3 * N / 8)) * N + (J + (N / 8))] = _c_m3n1;
        if (((I + (3 * N / 8)) < N) && ((J + (N / 4)) < N))
            C[(I + (3 * N / 8)) * N + (J + (N / 4))] = _c_m3n2;
        if (((I + (3 * N / 8)) < N) && ((J + (3 * N / 8)) < N))
            C[(I + (3 * N / 8)) * N + (J + (3 * N / 8))] = _c_m3n3;
        if (((I + (3 * N / 8)) < N) && ((J + (N / 2)) < N))
            C[(I + (3 * N / 8)) * N + (J + (N / 2))] = _c_m3n4;
        if (((I + (3 * N / 8)) < N) && ((J + (5 * N / 8)) < N))
            C[(I + (3 * N / 8)) * N + (J + (5 * N / 8))] = _c_m3n5;
        if (((I + (3 * N / 8)) < N) && ((J + (3 * N / 4)) < N))
            C[(I + (3 * N / 8)) * N + (J + (3 * N / 4))] = _c_m3n6;
        if (((I + (3 * N / 8)) < N) && ((J + (7 * N / 8)) < N))
            C[(I + (3 * N / 8)) * N + (J + (7 * N / 8))] = _c_m3n7;

        if (((I + (N / 2)) < N) && (J < N))
            C[(I + (N / 2)) * N + J] = _c_m4n0;
        if (((I + (N / 2)) < N) && ((J + (N / 8)) < N))
            C[(I + (N / 2)) * N + (J + (N / 8))] = _c_m4n1;
        if (((I + (N / 2)) < N) && ((J + (N / 4)) < N))
            C[(I + (N / 2)) * N + (J + (N / 4))] = _c_m4n2;
        if (((I + (N / 2)) < N) && ((J + (3 * N / 8)) < N))
            C[(I + (N / 2)) * N + (J + (3 * N / 8))] = _c_m4n3;
        if (((I + (N / 2)) < N) && ((J + (N / 2)) < N))
            C[(I + (N / 2)) * N + (J + (N / 2))] = _c_m4n4;
        if (((I + (N / 2)) < N) && ((J + (5 * N / 8)) < N))
            C[(I + (N / 2)) * N + (J + (5 * N / 8))] = _c_m4n5;
        if (((I + (N / 2)) < N) && ((J + (3 * N / 4)) < N))
            C[(I + (N / 2)) * N + (J + (3 * N / 4))] = _c_m4n6;
        if (((I + (N / 2)) < N) && ((J + (7 * N / 8)) < N))
            C[(I + (N / 2)) * N + (J + (7 * N / 8))] = _c_m4n7;

        if (((I + (5 * N / 8)) < N) && (J < N))
            C[(I + (5 * N / 8)) * N + J] = _c_m5n0;
        if (((I + (5 * N / 8)) < N) && ((J + (N / 8)) < N))
            C[(I + (5 * N / 8)) * N + (J + (N / 8))] = _c_m5n1;
        if (((I + (5 * N / 8)) < N) && ((J + (N / 4)) < N))
            C[(I + (5 * N / 8)) * N + (J + (N / 4))] = _c_m5n2;
        if (((I + (5 * N / 8)) < N) && ((J + (3 * N / 8)) < N))
            C[(I + (5 * N / 8)) * N + (J + (3 * N / 8))] = _c_m5n3;
        if (((I + (5 * N / 8)) < N) && ((J + (N / 2)) < N))
            C[(I + (5 * N / 8)) * N + (J + (N / 2))] = _c_m5n4;
        if (((I + (5 * N / 8)) < N) && ((J + (5 * N / 8)) < N))
            C[(I + (5 * N / 8)) * N + (J + (5 * N / 8))] = _c_m5n5;
        if (((I + (5 * N / 8)) < N) && ((J + (3 * N / 4)) < N))
            C[(I + (5 * N / 8)) * N + (J + (3 * N / 4))] = _c_m5n6;
        if (((I + (5 * N / 8)) < N) && ((J + (7 * N / 8)) < N))
            C[(I + (5 * N / 8)) * N + (J + (7 * N / 8))] = _c_m5n7;

        if (((I + (3 * N / 4)) < N) && (J < N))
            C[(I + (3 * N / 4)) * N + J] = _c_m6n0;
        if (((I + (5 * N / 8)) < N) && ((J + (N / 8)) < N))
            C[(I + (3 * N / 4)) * N + (J + (N / 8))] = _c_m6n1;
        if (((I + (5 * N / 8)) < N) && ((J + (N / 4)) < N))
            C[(I + (3 * N / 4)) * N + (J + (N / 4))] = _c_m6n2;
        if (((I + (5 * N / 8)) < N) && ((J + (3 * N / 8)) < N))
            C[(I + (3 * N / 4)) * N + (J + (3 * N / 8))] = _c_m6n3;
        if (((I + (5 * N / 8)) < N) && ((J + (N / 2)) < N))
            C[(I + (3 * N / 4)) * N + (J + (N / 2))] = _c_m6n4;
        if (((I + (5 * N / 8)) < N) && ((J + (5 * N / 8)) < N))
            C[(I + (3 * N / 4)) * N + (J + (5 * N / 8))] = _c_m6n5;
        if (((I + (5 * N / 8)) < N) && ((J + (3 * N / 4)) < N))
            C[(I + (3 * N / 4)) * N + (J + (3 * N / 4))] = _c_m6n6;
        if (((I + (5 * N / 8)) < N) && ((J + (7 * N / 8)) < N))
            C[(I + (3 * N / 4)) * N + (J + (7 * N / 8))] = _c_m6n7;

        if (((I + (7 * N / 8)) < N) && (J < N))
            C[(I + (7 * N / 8)) * N + J] = _c_m7n0;
        if (((I + (7 * N / 8)) < N) && ((J + (N / 8)) < N))
            C[(I + (7 * N / 8)) * N + (J + (N / 8))] = _c_m7n1;
        if (((I + (7 * N / 8)) < N) && ((J + (N / 4)) < N))
            C[(I + (7 * N / 8)) * N + (J + (N / 4))] = _c_m7n2;
        if (((I + (7 * N / 8)) < N) && ((J + (3 * N / 8)) < N))
            C[(I + (7 * N / 8)) * N + (J + (3 * N / 8))] = _c_m7n3;
        if (((I + (7 * N / 8)) < N) && ((J + (N / 2)) < N))
            C[(I + (7 * N / 8)) * N + (J + (N / 2))] = _c_m7n4;
        if (((I + (7 * N / 8)) < N) && ((J + (5 * N / 8)) < N))
            C[(I + (7 * N / 8)) * N + (J + (5 * N / 8))] = _c_m7n5;
        if (((I + (7 * N / 8)) < N) && ((J + (3 * N / 4)) < N))
            C[(I + (7 * N / 8)) * N + (J + (3 * N / 4))] = _c_m7n6;
        if (((I + (7 * N / 8)) < N) && ((J + (7 * N / 8)) < N))
            C[(I + (7 * N / 8)) * N + (J + (7 * N / 8))] = _c_m7n7;
        // printf("Result block_x %d thread_x %d block_y %d thread_y %d C[%d] %.6f C[%d] %.6f \n",block_x,thread_x,block_y,thread_y, I * N + J,C[I * N + J],I * N + (J+2),C[I * N + (J+2)]);
    }
}
#endif

// #################################################################

__global__ void matMul_1x1(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B)
{
    extern __shared__ _FTYPE_ As[];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim_bx = blockDim.x;
    int dim_by = blockDim.y;
    int TW = TILEDIM_K; //  each tile t has width TW

    int I = by * dim_by + ty; // Row
    int J = bx * dim_bx + tx; // Col

    _FTYPE_ _c = 0;
    for (unsigned int tile = 0; tile < ceil(N / (float)TW); tile++)
    {

        if ((I < N) && (tile * TW + tx) < N)
            As[ty * dim_by + tx] = A[I * N + tile * TW + tx];
        else
            As[ty * dim_by + tx] = 0.0f;

        if ((tile * TW + ty) < N && (J < N))
            As[dim_bx * dim_by + ty * dim_by + tx] = B[(tile * TW + ty) * N + J];
        else
            As[dim_bx * dim_by + ty * dim_by + tx] = 0.0f;

        __syncthreads();

        for (unsigned int k = 0; k < TW; k++)
        {
            _c += As[ty * dim_by + k] * As[dim_bx * dim_by + k * dim_by + tx];
        }
        __syncthreads();

        if ((I < N) && (J < N))
            C[I * N + J] = _c;
    }
}

__global__ void matMul_1x1_v1(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B)
{

    extern __shared__ _FTYPE_ As[];
    // extern __shared__ _FTYPE_ Bs[];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim_bx = blockDim.x;
    int dim_by = blockDim.y;

    //  each tile t has width T
    int TW = TILEDIM_K;

    int I = by * dim_by + ty;
    int J = bx * dim_bx + tx;

    if ((I < N) && (J < N))
    {
        _FTYPE_ _c = 0;
        for (unsigned int tile = 0; tile < ceil(N / (float)TW); tile++)
        {
            As[ty * dim_by + tx] = A[I * N + tile * TW + tx];

            As[dim_bx * dim_by + ty * dim_by + tx] = B[(tile * TW + ty) * N + J];
            __syncthreads();

            for (unsigned int k = 0; k < TW; k++)
            {
                _c += As[ty * dim_by + k] * As[dim_bx * dim_by + k * dim_by + tx];
            }
            __syncthreads();

            C[I * N + J] = _c;
        }
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

//  tile 4x4
__global__ void matMul_4x4(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B)
{

    extern __shared__ _FTYPE_ As[];
    // extern __shared__ _FTYPE_ Bs[];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim_bx = blockDim.x;
    int dim_by = blockDim.y;

    int I = by * dim_by + ty;
    int J = bx * dim_bx + tx;
    int TW = TILEDIM_K;

    if ((I < (N / 4)) && (J < (N / 4)))
    {
        _FTYPE_ _c_m0n0 = 0;
        _FTYPE_ _c_m0n1 = 0;
        _FTYPE_ _c_m0n2 = 0;
        _FTYPE_ _c_m0n3 = 0;

        _FTYPE_ _c_m1n0 = 0;
        _FTYPE_ _c_m1n1 = 0;
        _FTYPE_ _c_m1n2 = 0;
        _FTYPE_ _c_m1n3 = 0;

        _FTYPE_ _c_m2n0 = 0;
        _FTYPE_ _c_m2n1 = 0;
        _FTYPE_ _c_m2n2 = 0;
        _FTYPE_ _c_m2n3 = 0;

        _FTYPE_ _c_m3n0 = 0;
        _FTYPE_ _c_m3n1 = 0;
        _FTYPE_ _c_m3n2 = 0;
        _FTYPE_ _c_m3n3 = 0;

        for (unsigned int tile = 0; tile < N / TW; tile++)
        {
            As[ty * dim_by + tx] = A[I * N + tile * TW + tx];
            As[dim_bx * dim_by + ty * dim_by + tx] = A[(I + (N / 4)) * N + tile * TW + tx];
            As[2 * dim_bx * dim_by + ty * dim_by + tx] = A[(I + (N / 2)) * N + tile * TW + tx];
            As[3 * dim_bx * dim_by + ty * dim_by + tx] = A[(I + (3 * N / 4)) * N + tile * TW + tx];

            As[4 * dim_bx * dim_by + ty * dim_by + tx] = B[(tile * TW + ty) * N + J];
            As[5 * dim_bx * dim_by + ty * dim_by + tx] = B[(tile * TW + ty) * N + J + (N / 4)];
            As[6 * dim_bx * dim_by + ty * dim_by + tx] = B[(tile * TW + ty) * N + J + (N / 2)];
            As[7 * dim_bx * dim_by + ty * dim_by + tx] = B[(tile * TW + ty) * N + J + (3 * N / 4)];
            __syncthreads();

            for (unsigned int k = 0; k < TW; k++)
            {
                _c_m0n0 += As[ty * dim_by + k] * As[4 * dim_bx * dim_by + k * dim_by + tx];
                _c_m0n1 += As[ty * dim_by + k] * As[5 * dim_bx * dim_by + k * dim_by + tx];
                _c_m0n2 += As[ty * dim_by + k] * As[6 * dim_bx * dim_by + k * dim_by + tx];
                _c_m0n3 += As[ty * dim_by + k] * As[7 * dim_bx * dim_by + k * dim_by + tx];

                _c_m1n0 += As[dim_bx * dim_by + ty * dim_by + k] * As[4 * dim_bx * dim_by + k * dim_by + tx];
                _c_m1n1 += As[dim_bx * dim_by + ty * dim_by + k] * As[5 * dim_bx * dim_by + k * dim_by + tx];
                _c_m1n2 += As[dim_bx * dim_by + ty * dim_by + k] * As[6 * dim_bx * dim_by + k * dim_by + tx];
                _c_m1n3 += As[dim_bx * dim_by + ty * dim_by + k] * As[7 * dim_bx * dim_by + k * dim_by + tx];

                _c_m2n0 += As[2 * dim_bx * dim_by + ty * dim_by + k] * As[4 * dim_bx * dim_by + k * dim_by + tx];
                _c_m2n1 += As[2 * dim_bx * dim_by + ty * dim_by + k] * As[5 * dim_bx * dim_by + k * dim_by + tx];
                _c_m2n2 += As[2 * dim_bx * dim_by + ty * dim_by + k] * As[6 * dim_bx * dim_by + k * dim_by + tx];
                _c_m2n3 += As[2 * dim_bx * dim_by + ty * dim_by + k] * As[7 * dim_bx * dim_by + k * dim_by + tx];

                _c_m3n0 += As[3 * dim_bx * dim_by + ty * dim_by + k] * As[4 * dim_bx * dim_by + k * dim_by + tx];
                _c_m3n1 += As[3 * dim_bx * dim_by + ty * dim_by + k] * As[5 * dim_bx * dim_by + k * dim_by + tx];
                _c_m3n2 += As[3 * dim_bx * dim_by + ty * dim_by + k] * As[6 * dim_bx * dim_by + k * dim_by + tx];
                _c_m3n3 += As[3 * dim_bx * dim_by + ty * dim_by + k] * As[7 * dim_bx * dim_by + k * dim_by + tx];
            }
            __syncthreads();
        }
        C[I * N + J] = _c_m0n0;
        C[I * N + (J + (N / 4))] = _c_m0n1;
        C[I * N + (J + (N / 2))] = _c_m0n2;
        C[I * N + (J + (3 * N / 4))] = _c_m0n3;

        C[(I + (N / 4)) * N + J] = _c_m1n0;
        C[(I + (N / 4)) * N + (J + (N / 4))] = _c_m1n1;
        C[(I + (N / 4)) * N + (J + (N / 2))] = _c_m1n2;
        C[(I + (N / 4)) * N + (J + (3 * N / 4))] = _c_m1n3;

        C[(I + (N / 2)) * N + J] = _c_m2n0;
        C[(I + (N / 2)) * N + (J + (N / 4))] = _c_m2n1;
        C[(I + (N / 2)) * N + (J + (N / 2))] = _c_m2n2;
        C[(I + (N / 2)) * N + (J + (3 * N / 4))] = _c_m2n3;

        C[(I + (3 * N / 4)) * N + J] = _c_m3n0;
        C[(I + (3 * N / 4)) * N + (J + (N / 4))] = _c_m3n1;
        C[(I + (3 * N / 4)) * N + (J + (N / 2))] = _c_m3n2;
        C[(I + (3 * N / 4)) * N + (J + (3 * N / 4))] = _c_m3n3;
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

__global__ void matMul_old(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B)
{

    extern __shared__ _FTYPE_ As[];
    // extern __shared__ _FTYPE_ Bs[];

    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    int block_dim_x = blockDim.x;
    int block_dim_y = blockDim.y;

    int I = block_y * block_dim_y + thread_y;
    int J = block_x * block_dim_x + thread_x;

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

    // printf("block_x %d thread_x %d block_y %d thread_y %d I %d J %d \n",block_x,thread_x,block_y,thread_y,I,J);
    // printf("Thread Dim x %d y %d Block Dim %d %d \n",block_dim_x,block_dim_y,block_dim_x,block_dim_y);
    if ((I < (N / 8)) && (J < (N / 8)))
    {

        for (unsigned int c_tile = 0; c_tile < N / TILEDIM_K; c_tile++)
        {
            As[thread_y * block_dim_y + thread_x] = A[I * N + c_tile * TILEDIM_K + thread_x];
            As[block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = A[(I + (N / 8)) * N + c_tile * TILEDIM_K + thread_x];
            As[2 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = A[(I + (N / 4)) * N + c_tile * TILEDIM_K + thread_x];
            As[3 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = A[(I + (3 * N / 8)) * N + c_tile * TILEDIM_K + thread_x];
            As[4 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = A[(I + (N / 2)) * N + c_tile * TILEDIM_K + thread_x];
            As[5 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = A[(I + (5 * N / 8)) * N + c_tile * TILEDIM_K + thread_x];
            As[6 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = A[(I + (3 * N / 4)) * N + c_tile * TILEDIM_K + thread_x];
            As[7 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = A[(I + (7 * N / 8)) * N + c_tile * TILEDIM_K + thread_x];

            As[8 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = B[(c_tile * TILEDIM_K + thread_y) * N + J];
            As[9 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = B[(c_tile * TILEDIM_K + thread_y) * N + J + (N / 8)];
            As[10 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = B[(c_tile * TILEDIM_K + thread_y) * N + J + (N / 4)];
            As[11 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = B[(c_tile * TILEDIM_K + thread_y) * N + J + (3 * N / 8)];
            As[12 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = B[(c_tile * TILEDIM_K + thread_y) * N + J + (N / 2)];
            As[13 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = B[(c_tile * TILEDIM_K + thread_y) * N + J + (5 * N / 8)];
            As[14 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = B[(c_tile * TILEDIM_K + thread_y) * N + J + (3 * N / 4)];
            As[15 * block_dim_x * block_dim_y + thread_y * block_dim_y + thread_x] = B[(c_tile * TILEDIM_K + thread_y) * N + J + (7 * N / 8)];
            __syncthreads();

            // printf("before block_x %d block_y %d thread_x %d thread_y %d I %d J %d c_tile %d _c_m0n0 %f _c_m0n1 %f As[%d] %f A[%d] %f Bs[%d] %f B[%d] %f Bs_1[%d] %f B[%d] %f\n",block_x,block_y,thread_x,thread_y,I,J,c_tile, _c_m0n0,_c_m0n1,thread_y*block_dim_y+thread_x,As[thread_y*block_dim_y+thread_x],I*N + c_tile*TILEDIM_K + thread_x,A[I*N + c_tile*TILEDIM_K + thread_x],block_dim_x*block_dim_y+thread_y*block_dim_y+thread_x, As[block_dim_x*block_dim_y+thread_y*block_dim_y+thread_x],(c_tile*TILEDIM_K + thread_y)*N + J,B[(c_tile*TILEDIM_K + thread_y)*N + J],2*block_dim_x*block_dim_y + thread_y*block_dim_y+thread_x,As[2*block_dim_x*block_dim_y + thread_y*block_dim_y+thread_x],(c_tile*TILEDIM_K + thread_y)*N + J + 2,B[(c_tile*TILEDIM_K + thread_y)*N + J + (N/2)]);

            for (unsigned int k = 0; k < TILEDIM_K; k++)
            {
                _c_m0n0 += As[thread_y * block_dim_y + k] * As[8 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m0n1 += As[thread_y * block_dim_y + k] * As[9 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m0n2 += As[thread_y * block_dim_y + k] * As[10 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m0n3 += As[thread_y * block_dim_y + k] * As[11 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m0n4 += As[thread_y * block_dim_y + k] * As[12 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m0n5 += As[thread_y * block_dim_y + k] * As[13 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m0n6 += As[thread_y * block_dim_y + k] * As[14 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m0n7 += As[thread_y * block_dim_y + k] * As[15 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];

                _c_m1n0 += As[block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[8 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m1n1 += As[block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[9 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m1n2 += As[block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[10 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m1n3 += As[block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[11 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m1n4 += As[block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[12 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m1n5 += As[block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[13 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m1n6 += As[block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[14 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m1n7 += As[block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[15 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];

                _c_m2n0 += As[2 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[8 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m2n1 += As[2 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[9 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m2n2 += As[2 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[10 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m2n3 += As[2 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[11 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m2n4 += As[2 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[12 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m2n5 += As[2 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[13 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m2n6 += As[2 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[14 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m2n7 += As[2 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[15 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];

                _c_m3n0 += As[3 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[8 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m3n1 += As[3 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[9 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m3n2 += As[3 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[10 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m3n3 += As[3 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[11 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m3n4 += As[3 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[12 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m3n5 += As[3 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[13 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m3n6 += As[3 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[14 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m3n7 += As[3 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[15 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];

                _c_m4n0 += As[4 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[8 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m4n1 += As[4 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[9 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m4n2 += As[4 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[10 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m4n3 += As[4 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[11 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m4n4 += As[4 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[12 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m4n5 += As[4 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[13 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m4n6 += As[4 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[14 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m4n7 += As[4 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[15 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];

                _c_m5n0 += As[5 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[8 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m5n1 += As[5 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[9 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m5n2 += As[5 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[10 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m5n3 += As[5 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[11 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m5n4 += As[5 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[12 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m5n5 += As[5 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[13 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m5n6 += As[5 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[14 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m5n7 += As[5 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[15 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];

                _c_m6n0 += As[6 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[8 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m6n1 += As[6 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[9 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m6n2 += As[6 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[10 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m6n3 += As[6 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[11 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m6n4 += As[6 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[12 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m6n5 += As[6 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[13 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m6n6 += As[6 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[14 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m6n7 += As[6 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[15 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];

                _c_m7n0 += As[7 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[8 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m7n1 += As[7 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[9 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m7n2 += As[7 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[10 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m7n3 += As[7 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[11 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m7n4 += As[7 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[12 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m7n5 += As[7 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[13 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m7n6 += As[7 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[14 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];
                _c_m7n7 += As[7 * block_dim_x * block_dim_y + thread_y * block_dim_y + k] * As[15 * block_dim_x * block_dim_y + k * block_dim_y + thread_x];

                // printf("block_x %d block_y %d thread_x %d thread_y %d I %d J %d c_tile %d k %d A[%d] %f B[%d] %f As[%d] %f Bs_0[%d] %f Bs_1[%d] %f\n",block_x,block_y,thread_x,thread_y,I,J,c_tile,k,I*N + c_tile*TILEDIM_K + thread_x,A[I*N + c_tile*TILEDIM_K + thread_x],(c_tile*TILEDIM_K + thread_y)*N + J,B[(c_tile*TILEDIM_K + thread_y)*N + J],thread_y*block_dim_y+k,As[thread_y*block_dim_y+k],block_dim_x*block_dim_y + k*block_dim_y+thread_x,As[(block_dim_x*block_dim_y) + k*block_dim_y+thread_x],block_dim_x*block_dim_y + k*block_dim_y+block_dim_x*block_dim_y+thread_x,As[block_dim_x*block_dim_y + k*block_dim_y+block_dim_x*block_dim_y+thread_x]);
            }
            __syncthreads();
            // printf("block_x %d block_y %d thread_x %d thread_y %d I %d J %d c_tile %d _c_m0n0 %f _c_m0n1 %f \n",block_x,block_y,thread_x,thread_y,I,J,c_tile, _c_m0n0,_c_m0n1);
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
        // printf("Result block_x %d thread_x %d block_y %d thread_y %d C[%d] %.6f C[%d] %.6f \n",block_x,thread_x,block_y,thread_y, I * N + J,C[I * N + J],I * N + (J+2),C[I * N + (J+2)]);
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

    _FTYPE_ _c_[TS][TS] = {0};

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
                        _c_[mi][ni] += As[mi * dim_bx * dim_by + ty * dim_bx + k] * As[(TS + ni) * dim_bx * dim_by + k * dim_bx + tx];
                        // printf("bx %d by %d tx %d ty %d row %d col %d _c[%d,%d] %f As[%d] %f As[%d] %f\n", bx, by, tx, ty, row, col,mi,ni,_c_[mi][ni],mi * dim_bx * dim_by + ty * dim_bx + k,As[mi * dim_bx * dim_by + ty * dim_bx + k],(TS + ni) * dim_bx * dim_by + k * dim_bx + tx,As[(TS + ni) * dim_bx * dim_by + k * dim_bx + tx]);
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
                // printf("bx %d by %d tx %d ty %d row %d col %d C[%d,%d] %f _c[%d,%d] %f \n", bx, by, tx, ty, row, col, row + mi * npad / TS, col + ni * npad / TS,C[(row + mi * npad / TS) * N + (col + ni * npad / TS)],mi,ni,_c_[mi][ni]);
            }
        }
    }
}
