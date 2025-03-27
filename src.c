#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Reading the input arguments
    if (argc < 10) {
        printf("[ERROR] Usage: %s <inputFile> <px> <py> <pz> <nx> <ny> <nz> <nc> <outputFile>\n", argv[0]);
        return 0;
    }
    char *inputFile = argv[1];  // input file
    int px = atoi(argv[2]);     // number of processes in each direction
    int py = atoi(argv[3]);
    int pz = atoi(argv[4]);
    int nx = atoi(argv[5]);     // number of grid points in each direction
    int ny = atoi(argv[6]);
    int nz = atoi(argv[7]); 
    int nc = atoi(argv[8]);     // number of time steps
    char *outputFile = argv[9]; // output file

    // Check if the number of processes is correct
    if (size != px * py * pz) {
        if (rank == 0) {
            printf("[ERROR] The number of processes must be equal to px * py * pz = %d but were given %d\n", px * py * pz, size);
        }
        return 0;
    }

    // Reading the input file globally in the rank 0 process and then distributing it to all processes
    if (rank == 0) {
        int N = nx * ny * nz;
        double arr[3][N];

        // READING 
        FILE *file = fopen(inputFile, "r");
        if (file == NULL) {
            printf("[ERROR] Could not open file %s\n", inputFile);
            return 0;
        }

        for (int i = 0; i < N; i++) {
            fscanf(file, "%lf %lf %lf", &arr[0][i], &arr[1][i], &arr[2][i]);
        }
        fclose(file);
        printf("[DEBUG] Read the input file successfully\n");
    }
    
    MPI_Finalize();
    return 0;
}