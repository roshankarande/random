/* ///////////////////////////////////////////////////////////////////// */
/*!
  \file
  \brief Subarray example.

  Proc #0 creates a large array Abig with NROWS rows and NCOLS columns.
  Proc #1 receives a subarray with nrows_sub rows and ncols_sub
  columns starting at starts[].

  \author A. Mignone (mignone@to.infn.it)
  \date   March 14, 2020
*/
/* ///////////////////////////////////////////////////////////////////// */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "tools.c"

#define NROWS 5
#define NCOLS 6

int main(int argc, char **argv)
{
    int i, j;
    int rank, size;
    int nrows_sub = 3;
    int ncols_sub = 2;
    int starts[2] = {1, 3};
    int subsizes[2] = {nrows_sub, ncols_sub};
    int bigsizes[2] = {NROWS, NCOLS};
    int **Abig;
    int **Asub;
    MPI_Datatype MPI_Subarr;

    /* --------------------------------------------------------
       0. Initialize the MPI execution environment,
          create subarray type
       -------------------------------------------------------- */

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Type_create_subarray(2, bigsizes, subsizes, starts,
                             MPI_ORDER_C, MPI_INT, &MPI_Subarr);
    MPI_Type_commit(&MPI_Subarr);

    if (size < 2)
    {
        if (rank == 0)
        {
            fprintf(stderr, "! Need at least 2  processors.\n");
        }
        MPI_Finalize();
        return 1;
    }

    /* --------------------------------------------------------
       1. Proc #0 creates the big array,
          Proc #1 receives the subarray.
       -------------------------------------------------------- */

    if (rank == 0)
    {
        Abig = Allocate_2DintArray(NROWS, NCOLS);

        /* -- 1a. Fill array -- */

        for (i = 0; i < NROWS; i++)
        {
            for (j = 0; j < NCOLS; j++)
            {
                Abig[i][j] = j + i * NCOLS;
            }
        }

        /* -- 1b. Show array -- */

        Show_2DintArray(Abig, NROWS, NCOLS, "Big array (proc #0):");

        MPI_Send(&(Abig[0][0]), 1, MPI_Subarr, 1, 123, MPI_COMM_WORLD);

        free(Abig[0]);
        free(Abig);
    }
    else if (rank == 1)
    {

        Asub = Allocate_2DintArray(nrows_sub, ncols_sub);

        MPI_Recv(&(Asub[0][0]), nrows_sub * ncols_sub, MPI_INT, 0,
                 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        Show_2DintArray(Asub, nrows_sub, ncols_sub, "Received subarray (proc #1):");

        free(Asub[0]);
        free(Asub);
    }

    MPI_Type_free(&MPI_Subarr);
    MPI_Finalize();
    return 0;
}