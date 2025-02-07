#!/bin/bash
#PBS -q gpu_dgx@pbs-m1.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=8:ngpus=1:mem=256gb:scratch_ssd=420gb
#PBS -N wav2vec2-bart_base_scratch_e50
singularity run --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch:24.10-py3.SIF /storage/brno2/home/xhorni20/dp_mit/scripts/run_training_vox_scratch.sh
