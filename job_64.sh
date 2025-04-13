#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=48
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --time=00:10:00         ## wall-clock time limit
#SBATCH --partition=standard    ## can be "standard" or "cpu"

echo `date`
# Run for 5 iterations to get a variety of results
for i in {1..5}
do
    mpirun -np 64  ./src data_64_64_64_3.bin.txt 4 4 4 64 64 64 3 output_64_64_64_3_64_$i.txt

    mpirun -np 64  ./src data_64_64_96_7.bin.txt 4 4 4 64 64 96 7 output_64_64_96_7_64_$i.txt
done

# Run for 1 iteration to get final result
mpirun -np 64 ./src data_64_64_64_3.bin.txt 4 4 4 64 64 64 3 output_64_64_64_3_64.txt

mpirun -np 64 ./src data_64_64_96_7.bin.txt 4 4 4 64 64 96 7 output_64_64_96_7_64.txt
echo `date`