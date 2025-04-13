#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --time=00:10:00         ## wall-clock time limit
#SBATCH --partition=standard    ## can be "standard" or "cpu"

echo `date`
# Run for 5 iterations to get a variety of results
for i in {1..5}
do
    mpirun -np 1  ./src data_64_64_64_3.bin.txt 1 1 1 64 64 64 3 output_64_64_64_3_1_$i.txt
    mpirun -np 8  ./src data_64_64_64_3.bin.txt 2 2 2 64 64 64 3 output_64_64_64_3_8_$i.txt
    mpirun -np 16 ./src data_64_64_64_3.bin.txt 4 2 2 64 64 64 3 output_64_64_64_3_16_$i.txt
    mpirun -np 32 ./src data_64_64_64_3.bin.txt 4 4 2 64 64 64 3 output_64_64_64_3_32_$i.txt

    mpirun -np 1  ./src data_64_64_96_7.bin.txt 1 1 1 64 64 96 7 output_64_64_96_7_1_$i.txt
    mpirun -np 8  ./src data_64_64_96_7.bin.txt 2 2 2 64 64 96 7 output_64_64_96_7_8_$i.txt
    mpirun -np 16 ./src data_64_64_96_7.bin.txt 4 2 2 64 64 96 7 output_64_64_96_7_16_$i.txt
    mpirun -np 32 ./src data_64_64_96_7.bin.txt 4 4 2 64 64 96 7 output_64_64_96_7_32_$i.txt
done

# Run for 1 iteration to get final result
mpirun -np 1  ./src data_64_64_64_3.bin.txt 1 1 1 64 64 64 3 output_64_64_64_3_1.txt
mpirun -np 8  ./src data_64_64_64_3.bin.txt 2 2 2 64 64 64 3 output_64_64_64_3_8.txt
mpirun -np 16 ./src data_64_64_64_3.bin.txt 4 2 2 64 64 64 3 output_64_64_64_3_16.txt
mpirun -np 32 ./src data_64_64_64_3.bin.txt 4 4 2 64 64 64 3 output_64_64_64_3_32.txt

mpirun -np 1  ./src data_64_64_96_7.bin.txt 1 1 1 64 64 96 7 output_64_64_96_7_1.txt
mpirun -np 8  ./src data_64_64_96_7.bin.txt 2 2 2 64 64 96 7 output_64_64_96_7_8.txt
mpirun -np 16 ./src data_64_64_96_7.bin.txt 4 2 2 64 64 96 7 output_64_64_96_7_16.txt
mpirun -np 32 ./src data_64_64_96_7.bin.txt 4 4 2 64 64 96 7 output_64_64_96_7_32.txt
echo `date`