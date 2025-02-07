# 0. Go to the repository root
cd /storage/brno2/home/xhorni20/dp_mit

# 1. Open an interactive Singularity shell and execute the commands
# singularity shell --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch:24.10-py3.SIF

# Copy dataset to the scratch ssd
SOURCE_DIR="/storage/brno2/home/xhorni20/dp_mit/preprocessed_dataset"
DESTINATION_DIR="$SCRATCHDIR"
mkdir -p "$DESTINATION_DIR"
rsync -ahW --inplace --info=progress2 "$SOURCE_DIR" "$DESTINATION_DIR"
echo "Dataset copied to: $DESTINATION_DIR"

# Use $SCRATCHDIR for hf cache
export HF_HOME=$SCRATCHDIR/.cache/huggingface
# export HF_HOME=/storage/brno2/home/xhorni20/.cache/huggingface
export OMP_NUM_THREADS=2

# TODO: doesnt work all in one script
python -m pip install poetry
python -m poetry run pip install --no-build-isolation flash-attn
python -m poetry install
python -m poetry run ./run_libri_pretrained_decoder.sh

# ./bin/sclite -r demo.ref.txt -h demo.hyp.txt -i spu_id -o sum pra stdout  