#!/bin/bash
#SBATCH -J kernels_brute
#SBATCH -o kernels_brute_%j.txt
#SBATCH -e kernels_brute_%j.err
#SBATCH -p skx-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=24:00:00
#SBATCH -A TG-EAR170019
#SBATCH --mail-user=hrmd@mit.edu
#SBATCH --mail-type=all
python3 kernels_brute.py input_kernels.txt all --parallel --all_g_switches
