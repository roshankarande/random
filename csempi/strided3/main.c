#include "mpi.h"
#include <stdio.h>

#define NROWS 4
#define NCOLS 7

#define IROW 2
#define JCOL 4

void print1d(int *a, int n, char *message)
{
    printf("-----%s----\n", message);
    for (int i = 0; i < n; i++)
    {
        printf("%2d ", *(a++));
    }
    printf("\n\n");
}

void print2d(int *a, int m, int n, char *message)
{
    printf("-----%s----\n", message);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%2d ", *(a + i * n + j));
        }
        printf("\n");
    }
    printf("\n\n");
}

int main(int argc, char **argv)
{
    int i, j, rank, size;
    int A[NROWS][NCOLS], row[NCOLS], col[NROWS];

    MPI_Datatype row_type, col_type; //	Declare	new	datatypes
    MPI_Request sendReqs[2], recvReqs[2];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Type_contiguous(NCOLS, MPI_INT, &row_type);       //	Create	row_type
    MPI_Type_vector(NROWS, 1, NCOLS, MPI_INT, &col_type); //	Create	col_type
    MPI_Type_commit(&row_type);
    MPI_Type_commit(&col_type);

    if (rank == 0)
    {
        for (i = 0; i < NROWS; i++)
            for (j = 0; j < NCOLS; j++)
                A[i][j] = NCOLS * i + j + 1;

        print2d(&A[0][0], NROWS, NCOLS, "matrix A");
    }

    int irow = IROW; //	Index	of	the	row we want	to	send
    int jcol = JCOL; //	Index	of	the	col	we want	to	send

    if (rank == 0)
    {
        printf("Sending row=%d, col=%d\n", irow, jcol);
        MPI_Isend(&(A[irow][0]), 1, row_type, 1, 10, MPI_COMM_WORLD, sendReqs);
        MPI_Isend(&(A[0][jcol]), 1, col_type, 1, 11, MPI_COMM_WORLD, sendReqs + 1);
        // MPI_Send(&(A[irow][0]), 1, row_type, 1, 10, MPI_COMM_WORLD);
        // MPI_Send(&(A[0][jcol]), 1, col_type, 1, 11, MPI_COMM_WORLD);
    }
    else
    {
        // MPI_Irecv(row, 1, row_type, 0, 10, MPI_COMM_WORLD, recvReqs);
        // MPI_Irecv(col, 1, col_type, 0, 11, MPI_COMM_WORLD, recvReqs + 1);
        // MPI_Waitall(2, recvReqs, MPI_STATUSES_IGNORE);

        MPI_Irecv(row, NCOLS, MPI_INT, 0, 10, MPI_COMM_WORLD, recvReqs);
        MPI_Irecv(col, NROWS, MPI_INT, 0, 11, MPI_COMM_WORLD, recvReqs + 1);
        MPI_Waitall(2, recvReqs, MPI_STATUSES_IGNORE);

        // MPI_Recv(row, 1, row_type, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // MPI_Recv(col, 1, col_type, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // MPI_Recv(row, NCOLS, MPI_INT, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // MPI_Recv(col, NROWS, MPI_INT, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        print1d(row, NCOLS, "Received Row");
        print1d(col, NROWS, "Received Col");
    }
    MPI_Type_free(&row_type);
    MPI_Type_free(&col_type);

    MPI_Finalize();
}
