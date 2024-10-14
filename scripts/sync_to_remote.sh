#!/bin/bash
# Execute the script from root directory where the dp_mit directory is located

# Variables
LOCAL_DIR="dp_mit"
REMOTE_USER="xhorni20"
REMOTE_HOST="skirit.ics.muni.cz"
REMOTE_DIR="/storage/brno2/home/xhorni20"
EXCLUDES=(".venv" ".git")

# Construct the rsync exclude options
EXCLUDE_OPTS=""
for EXCLUDE in "${EXCLUDES[@]}"; do
    EXCLUDE_OPTS+="--exclude=$EXCLUDE "
done

# Execute rsync with exclusion and overwrite existing files on remote server
rsync -av --delete $EXCLUDE_OPTS $LOCAL_DIR $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR

# Output message
echo "Sync complete. Excluded directories: ${EXCLUDES[*]}"
