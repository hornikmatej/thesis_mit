# TODO: submit this script for qsub execution on the remote server 

# 0. Go to the repository root
cd /storage/brno2/home/xhorni20/dp_mit

# Use $SCRATCHDIR for hf cache
# export HF_HOME=$SCRATCHDIR/.cache/huggingface

export HF_HOME=/storage/brno2/home/xhorni20/.cache/huggingface

# 1. Open an interactive Singularity shell and execute the commands
singularity shell --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch:24.04-py3.SIF <<EOF
    python -m pip install poetry
    python -m poetry install
EOF