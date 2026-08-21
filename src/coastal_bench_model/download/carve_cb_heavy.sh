#!/bin/bash

#SBATCH --job-name=cb_heavy_dl
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/orcd/home/002/alexis31/PhyCoFlow_demo/4_CoastalBench_Reconstruction/logs/dl_%j.log

set -euo pipefail

cd /orcd/home/002/alexis31/PhyCoFlow_demo/4_CoastalBench_Reconstruction
mkdir -p logs

module purge
module load miniforge
source activate PhyCoEnv

python carve_cb_heavy.py \
      --dates 20080115 20080215 20080315 20080415 \
              20080515 20080615 20080715 20080815 \
              20080915 20081015 20081115 20081215 \
              20090115 20090215 20090315 20090415 \
              20090515 20090615 20090715 20090815 \
              20090915 20091015 20091115 20091215 \
      --vars u v temp salt \
      --pool-dir /home/alexis31/orcd/pool/CoastalBench \
      --subset-name CB_2YearMonthly_uvtempsalt \
      --raw-dir /home/alexis31/orcd/scratch/CoastalBench-heavy \
      --no-download
