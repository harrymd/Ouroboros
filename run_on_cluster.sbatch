#!/bin/bash
#SBATCH -J kernels_brute
#SBATCH -o kernels_brute_%j.txt
#SBATCH -e kernels_brute_%j.err
#SBATCH -p skx-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=00:30:00
#SBATCH -A TG-EAR170019
#SBATCH --mail-user=hrmd@mit.edu
#SBATCH --mail-type=all
python3 kernels/kernels_brute.py inputs/input_kernels.txt ka --parallel
