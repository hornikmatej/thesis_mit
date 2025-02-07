#!/bin/bash
#PBS -q gpu_dgx@pbs-m1.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=16:ngpus=1:mem=256gb:scratch_ssd=420gb
#PBS -N wav2vec2-bart_base_pretrained_decoder_e200
singularity run --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch:24.10-py3.SIF /storage/brno2/home/xhorni20/dp_mit/scripts/run_training_pretrained_decoder.sh
