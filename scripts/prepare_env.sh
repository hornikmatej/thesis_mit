!/bin/bash
# TODO: submit this script for qsub execution on the remote server 
PBS -N wav2vec2-bart_base_scratch_e25
PBS -q gpu_dgx
PBS -l select=1:ncpus=1:mem=128gb:scratch_ssd=256gb:ngpus=1
PBS -l walltime=24:00:00
PBS -m ae
singularity run --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch:24.10-py3.SIF /storage/brno2/home/xhorni20/dp_mit/scripts/run_training.sh
