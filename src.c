#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"

void exchangeGhostLayers(double *ghostLayer, double *localArr, int ip, int jp, int kp, int px, int py, int pz, int snx, int sny, int snz, int direction, int sendFlag) {
    int currRank = ip + px * jp + px * py * kp, neighbourRank;
    MPI_Datatype newType;
    MPI_Request request;

    // Left
    if (direction == 1){
        neighbourRank = (ip - 1) + px * jp + px * py * kp; 
        if (sendFlag){  
            MPI_Type_vector(sny * snz, 1, snx, MPI_DOUBLE, &newType);
            MPI_Type_commit(&newType);
            MPI_Isend(localArr + 0, 1, newType, neighbourRank, neighbourRank, MPI_COMM_WORLD, &request);
        } else {
            MPI_Irecv(ghostLayer, sny * snz, MPI_DOUBLE, neighbourRank, currRank, MPI_COMM_WORLD, &request);
        }
    }

    // Right
    if (direction == 2){
        neighbourRank = (ip + 1) + px * jp + px * py * kp; 
        if (sendFlag){  
            MPI_Type_vector(sny * snz, 1, snx, MPI_DOUBLE, &newType);
            MPI_Type_commit(&newType);
            MPI_Isend(localArr + (snx - 1), 1, newType, neighbourRank, neighbourRank, MPI_COMM_WORLD, &request);
        } else {
            MPI_Irecv(ghostLayer, sny * snz, MPI_DOUBLE, neighbourRank, currRank, MPI_COMM_WORLD, &request);
        }
    }

    // Top 
    if (direction == 3){
        neighbourRank = ip + px * (jp - 1) + px * py * kp; 
        if (sendFlag){  
            MPI_Type_vector(snz, snx, snx * sny, MPI_DOUBLE, &newType);
            MPI_Type_commit(&newType);
            MPI_Isend(localArr + 0, 1, newType, neighbourRank, neighbourRank, MPI_COMM_WORLD, &request);
        } else {
            MPI_Irecv(ghostLayer, snx * snz, MPI_DOUBLE, neighbourRank, currRank, MPI_COMM_WORLD, &request);
        }
    }

    // Bottom
    if (direction == 4){
        neighbourRank = ip + px * (jp + 1) + px * py * kp; 
        if (sendFlag){  
            MPI_Type_vector(snz, snx, snx * sny, MPI_DOUBLE, &newType);
            MPI_Type_commit(&newType);
            MPI_Isend(localArr + (snx * sny - snx), 1, newType, neighbourRank, neighbourRank, MPI_COMM_WORLD, &request);
        } else {
            MPI_Irecv(ghostLayer, snx * snz, MPI_DOUBLE, neighbourRank, currRank, MPI_COMM_WORLD, &request);
        }
    }

    // Front
    if (direction == 5){
        neighbourRank = ip + px * jp + px * py * (kp - 1);
        if (sendFlag){  
            MPI_Type_vector(1, snx * sny, snx * sny * snz, MPI_DOUBLE, &newType);
            MPI_Type_commit(&newType);
            MPI_Isend(localArr + 0, 1, newType, neighbourRank, neighbourRank, MPI_COMM_WORLD, &request);
        } else {
            MPI_Irecv(ghostLayer, snx * sny, MPI_DOUBLE, neighbourRank, currRank, MPI_COMM_WORLD, &request);
        }
    }

    // Back
    if (direction == 6){
        neighbourRank = ip + px * jp + px * py * (kp + 1);
        if (sendFlag){  
            MPI_Type_vector(1, snx * sny, snx * sny * snz, MPI_DOUBLE, &newType);
            MPI_Type_commit(&newType);
            MPI_Isend(localArr + (snx * sny * snz - snx * sny), 1, newType, neighbourRank, neighbourRank, MPI_COMM_WORLD, &request);
        } else {
            MPI_Irecv(ghostLayer, snx * sny, MPI_DOUBLE, neighbourRank, currRank, MPI_COMM_WORLD, &request);
        }     
    }

    // Wait for the send / receive to complete
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    return;
}


int main(int argc, char *argv[])
{
    // Initialize MPI environment
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


    // subdomain sizes
    int snx = nx / px;
    int sny = ny / py;
    int snz = nz / pz;

    int N = nx * ny * nz;
    int P = px * py * pz;
    int localN = snx * sny * snz;

    // also get the current process's coordinates
    int ip = (rank % (px * py)) % px;
    int jp = (rank % (px * py)) / px;
    int kp = (rank / (px * py));

    // Initializing variables to keep a track of time taken in each part of execution
    double time_1, time_2, time_3, time_4;
    time_1 = MPI_Wtime();

    // READING AND DISTRIBUTION STRATEGY 1 - too naive :(
    /*
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
    */

    // READING AND DISTRIBUTION STRATEGY 2 - parallel I/O
    double *localArr[nc];
    for (int t = 0; t < nc; t++){
        localArr[t] = malloc(localN * sizeof(double));        
    }

    // define starting position of each process
    int startPos = (ip * snx * nc) + (jp * sny * nx * nc) + (kp * snz * nx * ny * nc);

    // define number of reads for each process and count of each read
    int numReads = sny * snz;
    int readCount = nc * snx;

    // start reading the file
    MPI_File fh;
    int offset = 0;
    double *readArr = malloc(readCount * numReads * sizeof(double));
    MPI_File_open(MPI_COMM_WORLD, inputFile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    for (int k = 0; k < numReads; k++){
        offset = (k % sny) * nx * nc + (k / sny) * nx * ny * nc;
        MPI_File_read_at(fh, (startPos + offset) * sizeof(double), readArr + k * readCount, readCount, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&fh);
    
    // convert the data in required format
    for (int i = 0; i < localN; i++){
        for (int t = 0; t < nc; t++){
            localArr[t][i] = readArr[i * nc + t];
        }
    }
    free(readArr);
    printf("[DEBUG] Read the input file successfully for process %d\n", rank);

    // Reading and Data Distribution ends here
    time_2 = MPI_Wtime();
    
    // Main Code starts here
    // Before starting the computation it is wise to exchange the ghost layers beforehand
    double *leftGhostLayer[nc], *rightGhostLayer[nc], *topGhostLayer[nc], *bottomGhostLayer[nc], *frontGhostLayer[nc], *backGhostLayer[nc];

    for (int t = 0; t < nc; t++){
        // sending the ghost layers
        if (ip != 0){
            leftGhostLayer[t] = malloc(sny * snz * sizeof(double));
            exchangeGhostLayers(leftGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 1, 1);
        }
        if (ip != px - 1){
            rightGhostLayer[t] = malloc(sny * snz * sizeof(double));
            exchangeGhostLayers(rightGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 2, 1);
        }
        if (jp != 0){
            topGhostLayer[t] = malloc(snx * snz * sizeof(double));
            exchangeGhostLayers(topGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 3, 1);
        }
        if (jp != py - 1){
            bottomGhostLayer[t] = malloc(snx * snz * sizeof(double));
            exchangeGhostLayers(bottomGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 4, 1);
        }
        if (kp != 0){
            frontGhostLayer[t] = malloc(snx * sny * sizeof(double));
            exchangeGhostLayers(frontGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 5, 1);
        }
        if (kp != pz - 1){
            backGhostLayer[t] = malloc(snx * sny * sizeof(double));
            exchangeGhostLayers(backGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 6, 1);
        }

        // receiving the ghost layers
        if (ip != 0){
            leftGhostLayer[t] = malloc(sny * snz * sizeof(double));
            exchangeGhostLayers(leftGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 1, 0);
        }
        if (ip != px - 1){
            rightGhostLayer[t] = malloc(sny * snz * sizeof(double));
            exchangeGhostLayers(rightGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 2, 0);
        }
        if (jp != 0){
            topGhostLayer[t] = malloc(snx * snz * sizeof(double));
            exchangeGhostLayers(topGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 3, 0);
        }
        if (jp != py - 1){
            bottomGhostLayer[t] = malloc(snx * snz * sizeof(double));
            exchangeGhostLayers(bottomGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 4, 0);
        }
        if (kp != 0){
            frontGhostLayer[t] = malloc(snx * sny * sizeof(double));
            exchangeGhostLayers(frontGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 5, 0);
        }
        if (kp != pz - 1){
            backGhostLayer[t] = malloc(snx * sny * sizeof(double));
            exchangeGhostLayers(backGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 6, 0);
        }
    }

    // Local and Global Minima Calculations 
    int sxi, syi, szi, isLocalMinima, isLocalMaxima;
    int localMinima[nc], localMaxima[nc];
    double globalMinima[nc], globalMaxima[nc];
    for (int t = 0; t < nc; t++){
        localMinima[t] = 0;
        localMaxima[t] = 0;
        globalMinima[t] = localArr[t][0];
        globalMaxima[t] = localArr[t][0];

        for (int i = 0; i < localN; i++){
            globalMaxima[t] = (localArr[t][i] > globalMaxima[t]) ? localArr[t][i] : globalMaxima[t];
            globalMinima[t] = (localArr[t][i] < globalMinima[t]) ? localArr[t][i] : globalMinima[t];
            isLocalMinima = 1;
            isLocalMaxima = 1;

            sxi = (i % (snx * sny)) % snx;
            syi = (i % (snx * sny)) / snx;
            szi = (i / (snx * sny));

            if (sxi != 0){
                isLocalMinima = (localArr[t][i] < localArr[t][i - 1]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > localArr[t][i - 1]) ? isLocalMaxima : 0;
            } else if (ip != 0){
                isLocalMinima = (localArr[t][i] < leftGhostLayer[t][sny * szi + syi]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > leftGhostLayer[t][sny * szi + syi]) ? isLocalMaxima : 0;
            }

            if (sxi != snx - 1){
                isLocalMinima = (localArr[t][i] < localArr[t][i + 1]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > localArr[t][i + 1]) ? isLocalMaxima : 0;
            } else if (ip != px - 1){
                isLocalMinima = (localArr[t][i] < rightGhostLayer[t][sny * szi + syi]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > rightGhostLayer[t][sny * szi + syi]) ? isLocalMaxima : 0;
            }

            if (syi != 0){
                isLocalMinima = (localArr[t][i] < localArr[t][i - snx]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > localArr[t][i - snx]) ? isLocalMaxima : 0;
            } else if (jp != 0){
                isLocalMinima = (localArr[t][i] < topGhostLayer[t][snx * szi + sxi]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > topGhostLayer[t][snx * szi + sxi]) ? isLocalMaxima : 0;
            }

            if (syi != sny - 1){
                isLocalMinima = (localArr[t][i] < localArr[t][i + snx]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > localArr[t][i + snx]) ? isLocalMaxima : 0;
            } else if (jp != py - 1){
                isLocalMinima = (localArr[t][i] < bottomGhostLayer[t][snx * szi + sxi]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > bottomGhostLayer[t][snx * szi + sxi]) ? isLocalMaxima : 0;
            }

            if (szi != 0){
                isLocalMinima = (localArr[t][i] < localArr[t][i - snx * sny]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > localArr[t][i - snx * sny]) ? isLocalMaxima : 0;
            } else if (kp != 0){
                isLocalMinima = (localArr[t][i] < frontGhostLayer[t][snx * syi + sxi]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > frontGhostLayer[t][snx * syi + sxi]) ? isLocalMaxima : 0;
            }

            if (szi != snz - 1){
                isLocalMinima = (localArr[t][i] < localArr[t][i + snx * sny]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > localArr[t][i + snx * sny]) ? isLocalMaxima : 0;
            } else if (kp != pz - 1){
                isLocalMinima = (localArr[t][i] < backGhostLayer[t][snx * syi + sxi]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > backGhostLayer[t][snx * syi + sxi]) ? isLocalMaxima : 0;
            }

            if (isLocalMinima){
                localMinima[t] += 1;
            }
            if (isLocalMaxima){
                localMaxima[t] += 1;
            }
        }
    }

    // Main Code ends here
    time_3 = MPI_Wtime();

    // Gathering the local and global extermas to a single process
    int totalLocalMinima[nc], totalLocalMaxima[nc];
    double globalGlobalMinima[nc], globalGlobalMaxima[nc];
    for (int t = 0; t < nc; t++){
        MPI_Reduce(&localMinima[t], &totalLocalMinima[t], 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMaxima[t], &totalLocalMaxima[t], 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&globalMinima[t], &globalGlobalMinima[t], 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&globalMaxima[t], &globalGlobalMaxima[t], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }

    // Total time ends here
    time_4 = MPI_Wtime();

    // Calculating the time taken for each part of the code
    double readTime = time_2 - time_1;
    double mainTime = time_3 - time_2;
    double totalTime = time_4 - time_1;
    double finalReadTime, finalMainTime, finalTotalTime;
    MPI_Reduce(&readTime, &finalReadTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mainTime, &finalMainTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&totalTime, &finalTotalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Writing the output file
    if (rank == 0) {
        FILE *file = fopen(outputFile, "w");
        if (file == NULL) {
            printf("[ERROR] Could not open file %s\n", outputFile);
            return 0;
        }

        // Printing number of local extremas
        for (int t = 0; t < nc; t++){
            if (t != 0)
                fprintf(file, ", ");
            fprintf(file, "(%d, %d)", totalLocalMinima[t], totalLocalMaxima[t]);
        }
        fprintf(file, "\n");

        // Printing global extremas
        for (int t = 0; t < nc; t++){
            if (t != 0)
                fprintf(file, ", ");
            fprintf(file, "(%lf, %lf)", globalGlobalMinima[t], globalGlobalMaxima[t]);
        }
        fprintf(file, "\n");

        // Printing time taken
        fprintf(file, "%lf, %lf, %lf\n", finalReadTime, finalMainTime, finalTotalTime);

        fclose(file);
        printf("[DEBUG] Wrote the output file successfully\n");
    }
    
    MPI_Finalize();
    return 0;
}