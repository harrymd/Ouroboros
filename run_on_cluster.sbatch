#!/bin/bash
#SBATCH -J kernels_brute
#SBATCH -o kernels_brute_%j.txt
#SBATCH -e kernels_brute_%j.err
#SBATCH -p skx-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=04:00:00
#SBATCH -A TG-EAR170019
#SBATCH --mail-user=hrmd@mit.edu
#SBATCH --mail-type=all
export PYTHONPATH=$PYTHONPATH:/work/06414/tg857131
python3 kernels/kernels_brute.py inputs/input_kernels.txt ka --parallel
python3 kernels/kernels_brute.py inputs/input_kernels.txt mu --parallel
