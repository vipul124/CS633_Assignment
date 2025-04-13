#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"

void exchangeGhostLayers(float *ghostLayer, float *localArr, long long ip, long long jp, long long kp, long long px, long long py, long long pz, long long snx, long long sny, long long snz, long long direction, long long sendFlag)
{
    long long currRank = ip + px * jp + px * py * kp, neighbourRank;
    MPI_Datatype newType;
    MPI_Request request;

    // Left
    if (direction == 1)
    {
        neighbourRank = (ip - 1) + px * jp + px * py * kp;
        if (sendFlag)
        {
            MPI_Type_vector(sny * snz, 1, snx, MPI_FLOAT, &newType);
            MPI_Type_commit(&newType);
            MPI_Isend(localArr + 0, 1, newType, neighbourRank, neighbourRank, MPI_COMM_WORLD, &request);
        }
        else
        {
            MPI_Irecv(ghostLayer, sny * snz, MPI_FLOAT, neighbourRank, currRank, MPI_COMM_WORLD, &request);
        }
    }

    // Right
    if (direction == 2)
    {
        neighbourRank = (ip + 1) + px * jp + px * py * kp;
        if (sendFlag)
        {
            MPI_Type_vector(sny * snz, 1, snx, MPI_FLOAT, &newType);
            MPI_Type_commit(&newType);
            MPI_Isend(localArr + (snx - 1), 1, newType, neighbourRank, neighbourRank, MPI_COMM_WORLD, &request);
        }
        else
        {
            MPI_Irecv(ghostLayer, sny * snz, MPI_FLOAT, neighbourRank, currRank, MPI_COMM_WORLD, &request);
        }
    }

    // Top
    if (direction == 3)
    {
        neighbourRank = ip + px * (jp - 1) + px * py * kp;
        if (sendFlag)
        {
            MPI_Type_vector(snz, snx, snx * sny, MPI_FLOAT, &newType);
            MPI_Type_commit(&newType);
            MPI_Isend(localArr + 0, 1, newType, neighbourRank, neighbourRank, MPI_COMM_WORLD, &request);
        }
        else
        {
            MPI_Irecv(ghostLayer, snx * snz, MPI_FLOAT, neighbourRank, currRank, MPI_COMM_WORLD, &request);
        }
    }

    // Bottom
    if (direction == 4)
    {
        neighbourRank = ip + px * (jp + 1) + px * py * kp;
        if (sendFlag)
        {
            MPI_Type_vector(snz, snx, snx * sny, MPI_FLOAT, &newType);
            MPI_Type_commit(&newType);
            MPI_Isend(localArr + (snx * sny - snx), 1, newType, neighbourRank, neighbourRank, MPI_COMM_WORLD, &request);
        }
        else
        {
            MPI_Irecv(ghostLayer, snx * snz, MPI_FLOAT, neighbourRank, currRank, MPI_COMM_WORLD, &request);
        }
    }

    // Front
    if (direction == 5)
    {
        neighbourRank = ip + px * jp + px * py * (kp - 1);
        if (sendFlag)
        {
            MPI_Type_vector(1, snx * sny, snx * sny * snz, MPI_FLOAT, &newType);
            MPI_Type_commit(&newType);
            MPI_Isend(localArr + 0, 1, newType, neighbourRank, neighbourRank, MPI_COMM_WORLD, &request);
        }
        else
        {
            MPI_Irecv(ghostLayer, snx * sny, MPI_FLOAT, neighbourRank, currRank, MPI_COMM_WORLD, &request);
        }
    }

    // Back
    if (direction == 6)
    {
        neighbourRank = ip + px * jp + px * py * (kp + 1);
        if (sendFlag)
        {
            MPI_Type_vector(1, snx * sny, snx * sny * snz, MPI_FLOAT, &newType);
            MPI_Type_commit(&newType);
            MPI_Isend(localArr + (snx * sny * snz - snx * sny), 1, newType, neighbourRank, neighbourRank, MPI_COMM_WORLD, &request);
        }
        else
        {
            MPI_Irecv(ghostLayer, snx * sny, MPI_FLOAT, neighbourRank, currRank, MPI_COMM_WORLD, &request);
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
    if (argc < 10)
    {
        printf("[ERROR] Usage: %s <inputFile> <px> <py> <pz> <nx> <ny> <nz> <nc> <outputFile>\n", argv[0]);
        return 0;
    }
    char *inputFile = argv[1];     // input file
    long long px = atoll(argv[2]); // number of processes in each direction
    long long py = atoll(argv[3]);
    long long pz = atoll(argv[4]);
    long long nx = atoll(argv[5]); // number of grid points in each direction
    long long ny = atoll(argv[6]);
    long long nz = atoll(argv[7]);
    long long nc = atoll(argv[8]); // number of time steps
    char *outputFile = argv[9];    // output file

    // Check if the number of processes is correct
    if (size != px * py * pz)
    {
        if (rank == 0)
        {
            printf("[ERROR] The number of processes must be equal to px * py * pz = %lld but were given %d\n", px * py * pz, size);
        }
        return 0;
    }

    // subdomain sizes
    long long snx = nx / px;
    long long sny = ny / py;
    long long snz = nz / pz;

    long long N = nx * ny * nz;
    long long P = px * py * pz;
    long long localN = snx * sny * snz;

    // also get the current process's coordinates
    long long ip = (rank % (px * py)) % px;
    long long jp = (rank % (px * py)) / px;
    long long kp = (rank / (px * py));

    // Initializing variables to keep a track of time taken in each part of execution
    double time_1, time_2, time_3;
    long long localMinima[nc], localMaxima[nc], totalLocalMinima[nc], totalLocalMaxima[nc];
    float globalMinima[nc], globalMaxima[nc], globalGlobalMinima[nc], globalGlobalMaxima[nc];

    // Read Time starts here
    time_1 = MPI_Wtime();

    // READING AND DISTRIBUTION STRATEGY 1 - too naive :(
    /*
    // Reading the input file globally in the rank 0 process and then distributing it to all processes
    float *arr[nc];
    if (rank == 0) {
        for (long long t = 0; t < nc; t++){
            arr[t] = malloc(N * sizeof(float));
        }

        // READING STRATEGY - we are storing the input directly in distributable format
        FILE *file = fopen(inputFile, "rb");
        if (file == NULL) {
            printf("[ERROR] Could not open file %s\n", inputFile);
            return 0;
        }

        long long xi, yi, zi, sxi, syi, szi, si, pxi, pyi, pzi, pi, offset, i_mod;
        for (long long i = 0; i < N; i++) {
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

            for (long long t = 0; t < nc; t++){
                fread(&arr[t][i_mod], sizeof(float), 1, file);
            }
        }
        fclose(file);
    }

    // DISTRIBUTING STRATEGY - As we have already read the data in specific format we will directly distribute the data using MPI_Scatter
    float *localArr[nc];
    for (long long t = 0; t < nc; t++){
        localArr[t] = malloc(localN * sizeof(float));
    }
    for (long long t = 0; t < nc; t++){
        MPI_Scatter(arr[t], localN, MPI_FLOAT, localArr[t], localN, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    // if (rank == 0){
    //     for (long long t = 0; t < nc; t++){
    //         free(arr[t]);
    //     }
    // }
    */

    // READING AND DISTRIBUTION STRATEGY 2 - parallel I/O
    /*
    float *localArr[nc];
    for (long long t = 0; t < nc; t++)
    {
        localArr[t] = malloc(localN * sizeof(float));
    }

    // define starting position of each process
    long long startPos = (ip * snx * nc) + (jp * sny * nx * nc) + (kp * snz * nx * ny * nc);

    // define number of reads for each process and count of each read
    long long numReads = sny * snz;
    long long readCount = nc * snx;

    // start reading the file
    MPI_File fh;
    long long offset = 0;
    float *readArr = malloc(readCount * numReads * sizeof(float));
    MPI_File_open(MPI_COMM_WORLD, inputFile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    for (long long k = 0; k < numReads; k++)
    {
        offset = (k % sny) * nx * nc + (k / sny) * nx * ny * nc;
        MPI_File_read_at(fh, (startPos + offset) * sizeof(float), readArr + k * readCount, readCount, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&fh);

    // convert the data in required format
    for (long long i = 0; i < localN; i++)
    {
        for (long long t = 0; t < nc; t++)
        {
            localArr[t][i] = readArr[i * nc + t];
        }
    }
    // free(readArr);
    */

    // READING AND DISTRIBUTION STRATEGY 3 - parallel I/O w/ MPI_Type_vector
    float *localArr[nc];
    for (long long t = 0; t < nc; t++)
    {
        localArr[t] = malloc(localN * sizeof(float));
    }

    // define starting position of each process
    long long startPos = (ip * snx * nc) + (jp * sny * nx * nc) + (kp * snz * nx * ny * nc);

    // define number of reads for each process and count of each read
    long long numReads = sny * snz;
    long long readCount = nc * snx;

    // define blocklengths and displacements for MPI_Type_vector
    int *blocklengths = malloc(numReads * sizeof(int));
    int *displacements = malloc(numReads * sizeof(int));
    for (long long k = 0; k < numReads; k++)
    {
        blocklengths[k] = readCount;
        displacements[k] = ((k % sny) * nx * nc + (k / sny) * nx * ny * nc);
    }
    MPI_Datatype inputType;
    MPI_Type_indexed(numReads, blocklengths, displacements, MPI_FLOAT, &inputType);
    MPI_Type_commit(&inputType);

    // start reading the file
    MPI_File fh;
    float *readArr = malloc(readCount * numReads * sizeof(float));
    MPI_File_open(MPI_COMM_WORLD, inputFile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, startPos * sizeof(float), MPI_FLOAT, inputType, "native", MPI_INFO_NULL);
    MPI_File_read_all(fh, readArr, readCount * numReads, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    // convert the data in required format
    for (long long i = 0; i < localN; i++)
    {
        for (long long t = 0; t < nc; t++)
        {
            localArr[t][i] = readArr[i * nc + t];
        }
    }
    // free(readArr);

    // Reading and Data Distribution ends here
    MPI_Barrier(MPI_COMM_WORLD);
    time_2 = MPI_Wtime();

    // Main Code starts here
    // Before starting the computation it is wise to exchange the ghost layers beforehand
    float *leftGhostLayer[nc], *rightGhostLayer[nc], *topGhostLayer[nc], *bottomGhostLayer[nc], *frontGhostLayer[nc], *backGhostLayer[nc];

    for (long long t = 0; t < nc; t++)
    {
        // sending the ghost layers
        if (ip != 0)
        {
            leftGhostLayer[t] = malloc(sny * snz * sizeof(float));
            exchangeGhostLayers(leftGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 1, 1);
        }
        if (ip != px - 1)
        {
            rightGhostLayer[t] = malloc(sny * snz * sizeof(float));
            exchangeGhostLayers(rightGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 2, 1);
        }
        if (jp != 0)
        {
            topGhostLayer[t] = malloc(snx * snz * sizeof(float));
            exchangeGhostLayers(topGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 3, 1);
        }
        if (jp != py - 1)
        {
            bottomGhostLayer[t] = malloc(snx * snz * sizeof(float));
            exchangeGhostLayers(bottomGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 4, 1);
        }
        if (kp != 0)
        {
            frontGhostLayer[t] = malloc(snx * sny * sizeof(float));
            exchangeGhostLayers(frontGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 5, 1);
        }
        if (kp != pz - 1)
        {
            backGhostLayer[t] = malloc(snx * sny * sizeof(float));
            exchangeGhostLayers(backGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 6, 1);
        }

        // receiving the ghost layers
        if (ip != 0)
        {
            leftGhostLayer[t] = malloc(sny * snz * sizeof(float));
            exchangeGhostLayers(leftGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 1, 0);
        }
        if (ip != px - 1)
        {
            rightGhostLayer[t] = malloc(sny * snz * sizeof(float));
            exchangeGhostLayers(rightGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 2, 0);
        }
        if (jp != 0)
        {
            topGhostLayer[t] = malloc(snx * snz * sizeof(float));
            exchangeGhostLayers(topGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 3, 0);
        }
        if (jp != py - 1)
        {
            bottomGhostLayer[t] = malloc(snx * snz * sizeof(float));
            exchangeGhostLayers(bottomGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 4, 0);
        }
        if (kp != 0)
        {
            frontGhostLayer[t] = malloc(snx * sny * sizeof(float));
            exchangeGhostLayers(frontGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 5, 0);
        }
        if (kp != pz - 1)
        {
            backGhostLayer[t] = malloc(snx * sny * sizeof(float));
            exchangeGhostLayers(backGhostLayer[t], localArr[t], ip, jp, kp, px, py, pz, snx, sny, snz, 6, 0);
        }
    }

    // Local and Global Minima Calculations
    long long sxi, syi, szi, isLocalMinima, isLocalMaxima;
    for (long long t = 0; t < nc; t++)
    {
        localMinima[t] = 0;
        localMaxima[t] = 0;
        globalMinima[t] = localArr[t][0];
        globalMaxima[t] = localArr[t][0];

        for (long long i = 0; i < localN; i++)
        {
            globalMaxima[t] = (localArr[t][i] > globalMaxima[t]) ? localArr[t][i] : globalMaxima[t];
            globalMinima[t] = (localArr[t][i] < globalMinima[t]) ? localArr[t][i] : globalMinima[t];
            isLocalMinima = 1;
            isLocalMaxima = 1;

            sxi = (i % (snx * sny)) % snx;
            syi = (i % (snx * sny)) / snx;
            szi = (i / (snx * sny));

            if (sxi != 0)
            {
                isLocalMinima = (localArr[t][i] < localArr[t][i - 1]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > localArr[t][i - 1]) ? isLocalMaxima : 0;
            }
            else if (ip != 0)
            {
                isLocalMinima = (localArr[t][i] < leftGhostLayer[t][sny * szi + syi]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > leftGhostLayer[t][sny * szi + syi]) ? isLocalMaxima : 0;
            }

            if (sxi != snx - 1)
            {
                isLocalMinima = (localArr[t][i] < localArr[t][i + 1]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > localArr[t][i + 1]) ? isLocalMaxima : 0;
            }
            else if (ip != px - 1)
            {
                isLocalMinima = (localArr[t][i] < rightGhostLayer[t][sny * szi + syi]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > rightGhostLayer[t][sny * szi + syi]) ? isLocalMaxima : 0;
            }

            if (syi != 0)
            {
                isLocalMinima = (localArr[t][i] < localArr[t][i - snx]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > localArr[t][i - snx]) ? isLocalMaxima : 0;
            }
            else if (jp != 0)
            {
                isLocalMinima = (localArr[t][i] < topGhostLayer[t][snx * szi + sxi]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > topGhostLayer[t][snx * szi + sxi]) ? isLocalMaxima : 0;
            }

            if (syi != sny - 1)
            {
                isLocalMinima = (localArr[t][i] < localArr[t][i + snx]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > localArr[t][i + snx]) ? isLocalMaxima : 0;
            }
            else if (jp != py - 1)
            {
                isLocalMinima = (localArr[t][i] < bottomGhostLayer[t][snx * szi + sxi]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > bottomGhostLayer[t][snx * szi + sxi]) ? isLocalMaxima : 0;
            }

            if (szi != 0)
            {
                isLocalMinima = (localArr[t][i] < localArr[t][i - snx * sny]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > localArr[t][i - snx * sny]) ? isLocalMaxima : 0;
            }
            else if (kp != 0)
            {
                isLocalMinima = (localArr[t][i] < frontGhostLayer[t][snx * syi + sxi]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > frontGhostLayer[t][snx * syi + sxi]) ? isLocalMaxima : 0;
            }

            if (szi != snz - 1)
            {
                isLocalMinima = (localArr[t][i] < localArr[t][i + snx * sny]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > localArr[t][i + snx * sny]) ? isLocalMaxima : 0;
            }
            else if (kp != pz - 1)
            {
                isLocalMinima = (localArr[t][i] < backGhostLayer[t][snx * syi + sxi]) ? isLocalMinima : 0;
                isLocalMaxima = (localArr[t][i] > backGhostLayer[t][snx * syi + sxi]) ? isLocalMaxima : 0;
            }

            if (isLocalMinima)
            {
                localMinima[t] += 1;
            }
            if (isLocalMaxima)
            {
                localMaxima[t] += 1;
            }
        }
    }

    // Gathering the local and global extermas to a single process
    for (long long t = 0; t < nc; t++)
    {
        MPI_Reduce(&localMinima[t], &totalLocalMinima[t], 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMaxima[t], &totalLocalMaxima[t], 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&globalMinima[t], &globalGlobalMinima[t], 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&globalMaxima[t], &globalGlobalMaxima[t], 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    }

    // Total time & Main Code ends here
    time_3 = MPI_Wtime();

    // Calculating the time taken for each part of the code
    double readTime = time_2 - time_1;
    double mainTime = time_3 - time_2;
    double totalTime = time_3 - time_1;
    double finalReadTime, finalMainTime, finalTotalTime;
    MPI_Reduce(&readTime, &finalReadTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mainTime, &finalMainTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&totalTime, &finalTotalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Writing the output file
    if (rank == 0)
    {
        FILE *file = fopen(outputFile, "w");
        if (file == NULL)
        {
            printf("[ERROR] Could not open file %s\n", outputFile);
            return 0;
        }

        // Printing number of local extremas
        for (long long t = 0; t < nc; t++)
        {
            if (t != 0)
                fprintf(file, ", ");
            fprintf(file, "(%lld, %lld)", totalLocalMinima[t], totalLocalMaxima[t]);
        }
        fprintf(file, "\n");

        // Printing global extremas
        for (long long t = 0; t < nc; t++)
        {
            if (t != 0)
                fprintf(file, ", ");
            fprintf(file, "(%.4f, %.4f)", globalGlobalMinima[t], globalGlobalMaxima[t]);
        }
        fprintf(file, "\n");

        // Printing time taken
        fprintf(file, "%lf, %lf, %lf\n", finalReadTime, finalMainTime, finalTotalTime);

        fclose(file);
    }

    MPI_Finalize();
    return 0;
}