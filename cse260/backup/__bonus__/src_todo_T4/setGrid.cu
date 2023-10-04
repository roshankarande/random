
#include "mytypes.h"
#include <stdio.h>

void setGrid(int m, int n, dim3 &blockDim, dim3 &gridDim)
{

   // set your block dimensions and grid dimensions here
   gridDim.x = n / (TILESCALE * blockDim.x * TILEDIM_X);
   gridDim.y = m / (TILESCALE * blockDim.y * TILEDIM_Y);

   // you can overwrite blockDim here if you like.
   if (n % (TILESCALE * blockDim.x * TILEDIM_X) != 0)
      gridDim.x++;
   if (n % (TILESCALE * blockDim.y * TILEDIM_Y) != 0)
      gridDim.y++;
}
