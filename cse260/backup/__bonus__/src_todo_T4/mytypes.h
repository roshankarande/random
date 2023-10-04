#ifndef MYTYPES_H
#define MYTYPES_H

// ----------------------------------------------------------------------
// #define kernel matMul_8x8 // old -> explicit -> without checks
// #define kernel matMul   // old -> explicit -> with checks
//#define kernel matMul_naive
// #define kernel matmul_tilescale
// ----------------------------------------------------------------------

// matmul_2dtile_tilescale   ---> tilescaling + 2dtile
// matmul_2dtile  ---> 2dtile only
// matmul_tilescale   ---> tilescaling only

// NOTE!  TILEDIM_K has to be <= min(BLOCKDIM_X, BLOCKDIM_Y)
// --------------------------------------------------------------------------

// #define TILESCALE 4
// #define TILEDIM_X 3
// #define TILEDIM_Y 4

// #define TILEDIM_K 16
// #define BLOCKDIM_X 16
// #define BLOCKDIM_Y 16

// #define kernel matmul_2dtile_tilescale // new one

// -------------------------------------------------------------------------

// -------------------------------------------------------------------------

// Only 2d tiling
// #define TILESCALE 1
// #define TILEDIM_X 16
// #define TILEDIM_Y 16

// #define TILEDIM_K 16
// #define BLOCKDIM_X 16
// #define BLOCKDIM_Y 16

// #define kernel matmul_2dtile

// -------------------------------------------------------------------------

#define TILESCALE 8
#define TILEDIM_X 1
#define TILEDIM_Y 1
#define TILEDIM_K 16
#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16
#define kernel matmul_tilescale

// ---------------------------------------------------------------------------

// -------------------------------------------------------------------------

//#define TILEDIM_K TILEDIM // Enter your own values

// tilescale (# of points computed by each thread)
#ifndef TILESCALE_M
#define TILESCALE_M 1 // Enter your own values
#endif
#ifndef TILESCALE_N
#define TILESCALE_N 1 // Enter your own values
#endif
#ifndef TILESCALE_K
#define TILESCALE_K 1 // Enter your own values
#endif

//#define TILEDIM_M TILEDIM // Enter your own values
//#define TILEDIM_N TILEDIM // Enter your own values

// matrix A loads
// with warps along the horiziontal axis (K)
// so to get good coalescaed loads, we want TILEDIM_K to be >= 32
//

// step size in each dimension
#define TILESTEP_N 1 // Enter your own values
#define TILESTEP_K 1 // Enter your own values
#define TILESTEP_M 1 // Enter your own values

#endif
