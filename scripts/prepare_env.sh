#!/bin/bash
#PBS -q gpu_dgx@pbs-m1.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=1:ngpus=1:mem=192gb:scratch_ssd=256gb
#PBS -N wav2vec2-bart_base_scratch_e30
singularity run --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch:24.10-py3.SIF /storage/brno2/home/xhorni20/dp_mit/scripts/run_training.sh
