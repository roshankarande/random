#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[])
{
    int rank, size, i;
    int NCols = 4, NRows = 4;

    int buffer[NCols * NRows];
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (size < 2)
    {
        printf("Please run with 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    // colType ---> horizontal
    // rowType ---> vertical
    MPI_Datatype vertical, horizontal;

    MPI_Type_contiguous(NCols, MPI_DOUBLE, &horizontal);
    MPI_Type_commit(&horizontal);

    // MPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype, MPI_Datatype *new_type)
    MPI_Type_vector(NRows, 1, NCols, MPI_DOUBLE, &vertical);
    MPI_Type_commit(&vertical);

    MPI_Sendrecv(1, horizontal,);
    MPI_Sendrecv(1, vertical)

    if (rank == 0)
    {
        for (i = 0; i < 24; i++)
            buffer[i] = i;
        MPI_Send(buffer, 1, type, 1, 123, MPI_COMM_WORLD);
    }

    if (rank == 1)
    {
        for (i = 0; i < 24; i++)
            buffer[i] = -1;
        MPI_Recv(buffer, 1, type, 0, 123, MPI_COMM_WORLD, &status);
        for (i = 0; i < 24; i++)
            printf("buffer[%d] = %d\n", i, buffer[i]);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}

