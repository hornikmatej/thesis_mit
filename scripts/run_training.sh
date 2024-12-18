# 0. Go to the repository root
cd /storage/brno2/home/xhorni20/dp_mit
# Create the necessary directories in the scratch space
# mkdir -p $SCRATCHDIR/.cache/huggingface/datasets
# rsync -r --info=progress2 /storage/brno2/home/xhorni20/.cache/huggingface/datasets $SCRATCHDIR/.cache/huggingface

# 1. Open an interactive Singularity shell and execute the commands
# singularity shell --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch:24.10-py3.SIF

# Use $SCRATCHDIR for hf cache
# export HF_HOME=$SCRATCHDIR/.cache/huggingface
export HF_HOME=/storage/brno2/home/xhorni20/.cache/huggingface
export OMP_NUM_THREADS=2

# TODO: doesnt work all in one script
python -m pip install poetry
python -m poetry install
python -m poetry run ./run_libri.sh

# ./bin/sclite -r demo.ref.txt -h demo.hyp.txt -i spu_id -o sum pra stdout  