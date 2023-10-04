/* ///////////////////////////////////////////////////////////////////// */
/*!
  \file
  \brief Collection of useful functions for the MPI course.

  This file provides some simple function for memory allocation
  (dbl and int 2D array), printing.
  Function prototyping is already included here and should not be
  repeated elsewhere.
  Simply include this file after the main header section when needed:
  \code
  #include <stdio.h>
  ...
  #include "tools.c"
  ...
  int main()
  {
    .
    .
    .
  }
  \endcodes

  \author A. Mignone (mignone@to.infn.it)
  \date   March 14, 2020
*/
/* ///////////////////////////////////////////////////////////////////// */
double **Allocate_2DdblArray(int, int);
int **Allocate_2DintArray(int, int);
void Show_2DdblArray(double **, int, int, const char *);
void Show_2DintArray(int **, int, int, const char *);

/* ********************************************************************* */
double **Allocate_2DdblArray(int nx, int ny)
/*
 * Allocate memory for a double precision array with
 * nx rows and ny columns
 *********************************************************************** */
{
    int i, j;
    double **buf;

    buf = (double **)malloc(nx * sizeof(double *));
    buf[0] = (double *)malloc(nx * ny * sizeof(double));
    for (j = 1; j < nx; j++)
        buf[j] = buf[j - 1] + ny;

    return buf;
}
/* ********************************************************************* */
int **Allocate_2DintArray(int nx, int ny)
/*
 * Allocate memory for an integer-type array with
 * nx rows and ny columns
 *********************************************************************** */
{
    int i, j;
    int **buf;

    buf = (int **)malloc(nx * sizeof(int *));
    buf[0] = (int *)malloc(nx * ny * sizeof(int));
    for (j = 1; j < nx; j++)
        buf[j] = buf[j - 1] + ny;

    return buf;
}

/* ********************************************************************* */
void Show_2DdblArray(double **A, int nx, int ny, const char *string)
/*
 *********************************************************************** */
{
    int i, j;

    printf("%s\n", string);
    printf("------------------------------\n");
    for (i = 0; i < nx; i++)
    {
        for (j = 0; j < ny; j++)
        {
            printf("%8.2f  ", A[i][j]);
        }
        printf("\n");
    }
    printf("------------------------------\n");
}
/* ********************************************************************* */
void Show_2DintArray(int **A, int nx, int ny, const char *string)
/*
 *********************************************************************** */
{
    int i, j;

    printf("%s\n", string);
    for (j = 0; j < ny; j++)
        printf("-----");
    printf("\n");

    for (i = 0; i < nx; i++)
    {
        for (j = 0; j < ny; j++)
        {
            printf("%03d  ", A[i][j]);
        }
        printf("\n");
    }

    for (j = 0; j < ny; j++)
        printf("-----");
    printf("\n");
}
