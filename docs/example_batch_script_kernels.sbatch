#!/bin/bash
#SBATCH -J kernels_brute
#SBATCH -o kernels_brute_%j.txt
#SBATCH -e kernels_brute_%j.err
#SBATCH -p queue-name 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=00:01:00
#SBATCH -A allocation-name
#SBATCH --mail-user=user@domain.com
#SBATCH --mail-type=all
python3 kernels_brute.py input_kernels.txt all --parallel --all_g_switches
