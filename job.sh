#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --time=00:10:00         ## wall-clock time limit
#SBATCH --partition=standard    ## can be "standard" or "cpu"

echo `date`
mpirun -np 8 ./src data_64_64_64_3.txt 2 2 2 64 64 64 3 output_64_64_64_3_8.txt
mpirun -np 16 ./src data_64_64_64_3.txt 4 2 2 64 64 64 3 output_64_64_64_3_16.txt
mpirun -np 32 ./src data_64_64_64_3.txt 4 4 2 64 64 64 3 output_64_64_64_3_32.txt
mpirun -np 64 ./src data_64_64_64_3.txt 4 4 4 64 64 64 3 output_64_64_64_3_64.txt
mpirun -np 8 ./src data_64_64_96_7.txt 2 2 2 64 64 96 7 output_64_64_96_7_8.txt
mpirun -np 16 ./src data_64_64_96_7.txt 4 2 2 64 64 96 7 output_64_64_96_7_16.txt
mpirun -np 32 ./src data_64_64_96_7.txt 4 4 2 64 64 96 7 output_64_64_96_7_32.txt
mpirun -np 64 ./src data_64_64_96_7.txt 4 4 4 64 64 96 7 output_64_64_96_7_64.txt
echo `date`