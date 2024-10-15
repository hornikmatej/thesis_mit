# TODO: submit this script for qsub execution on the remote server 
#!/bin/bash
#PBS -N PyTorch_Job
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=16gb:scratch_local=10gb:ngpus=1:gpu_cap=cuda60
#PBS -l walltime=4:00:00
#PBS -m ae
# singularity run --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch:24.04-py3.SIF /your/work_dir/run_script.sh

# 0. Go to the repository root
cd /storage/brno2/home/xhorni20/dp_mit


# 1. Open an interactive Singularity shell and execute the commands
singularity shell --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch:24.04-py3.SIF

# Use $SCRATCHDIR for hf cache
export HF_HOME=$SCRATCHDIR/.cache/huggingface
# export HF_HOME=/storage/brno2/home/xhorni20/.cache/huggingface

# TODO: doesnt work all in one script
python -m pip install poetry
python -m poetry install

# #!/bin/bash
# # Make sure install_poetry.sh is executable
# chmod +x install_poetry.sh

# # Open an interactive Singularity shell and execute the script
# singularity shell --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch:24.04-py3.SIF <<EOF
#     ./install_poetry.sh
#     exec bash
# EOF

# #!/bin/bash
# python -m pip install poetry
# python -m poetry install