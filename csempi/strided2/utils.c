

void initBuffer(double *a, int n, int rank)
{
    for (int i = 0; i < n * n; i++)
    {
        *(a++) = rank * n * n + i;
    }
}