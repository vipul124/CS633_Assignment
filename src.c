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

    // subdomain sizes
    int snx = nx / px;
    int sny = ny / py;
    int snz = nz / pz;

    int N = nx * ny * nz;
    int localN = snx * sny * snz;

    // Check if the number of processes is correct
    if (size != px * py * pz) {
        if (rank == 0) {
            printf("[ERROR] The number of processes must be equal to px * py * pz = %d but were given %d\n", px * py * pz, size);
        }
        return 0;
    }

    // Reading the input file globally in the rank 0 process and then distributing it to all processes
    double *arr[nc];
    if (rank == 0) {
        for (int t = 0; t < nc; t++){
            arr[t] = malloc(N * sizeof(double));
        }
        
        // READING STRATEGY - we are storing the input directly in distributable format
        FILE *file = fopen(inputFile, "r");
        if (file == NULL) {
            printf("[ERROR] Could not open file %s\n", inputFile);
            return 0;
        }

        int xi, yi, zi, sxi, syi, szi, si, pxi, pyi, pzi, pi, offset, i_mod;
        for (int i = 0; i < N; i++) {
            // Get the coordinates in current domain
            xi = (i % (nx * ny)) % nx;
            yi = (i % (nx * ny)) / nx;
            zi = (i / (nx * ny));

            // Convert the coordinates to subdomain and Get the position using the subdomain coordinates
            sxi = xi % snx;
            syi = yi % sny;
            szi = zi % snz;
            si = sxi + snx * syi + snx * sny * szi;

            // Get the process coordinates and Get the process id using the process coordinates
            pxi = xi / snx;
            pyi = yi / sny;
            pzi = zi / snz;
            pi = pxi + px * pyi + px * py * pzi;

            // Calculate the offset and then the modified i
            offset = pi * snx * sny * snz;
            i_mod = si + offset;

            for (int t = 0; t < nc; t++){
                fscanf(file, "%lf", &arr[t][i_mod]);
            }
        }
        fclose(file);
        printf("[DEBUG] Read the input file successfully\n");
    }

    // DISTRIBUTING STRATEGY - As we have already read the data in specific format we will directly distribute the data using MPI_Scatter
    double *localArr[nc];
    for (int t = 0; t < nc; t++){
        localArr[t] = malloc(localN * sizeof(double));        
    }
    for (int t = 0; t < nc; t++){
        MPI_Scatter(arr[t], localN, MPI_DOUBLE, localArr[t], localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    printf("[DEBUG] Distributed input data to process %d\n", rank);
    if (rank == 0){
        for (int t = 0; t < nc; t++){
            free(arr[t]);
        }
    }
    
    MPI_Finalize();
    return 0;
}