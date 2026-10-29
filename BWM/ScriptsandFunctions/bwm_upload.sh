##!/bin/bash
#
## --- Configuration ---
#LOCAL_DIR="Data"   # Path to the directory containing subdirectories
#REMOTE_USER="cl1704"              # Your remote SSH username
#REMOTE_HOST="della.princeton.edu"         # Remote server IP or hostname
#REMOTE_TARGET="/scratch/gpfs/TENGEL/cl1704/embedding_dimension/BWM/Data"
## ---------------------
#
## Change to the local directory
#cd "$LOCAL_DIR" || { echo "Local directory not found"; exit 1; }
#
## Loop through all directories in the current folder
#for dir in */; do
#    # Remove the trailing slash from the directory name
#    dir=${dir%/}
#
#    echo "Checking if '$dir' exists on remote..."
#
#    # Check if the directory exists on the remote server
#    if ssh "${REMOTE_USER}@${REMOTE_HOST}" "[ -d '${REMOTE_TARGET}/${dir}' ]"; then
#        echo "--> '$dir' already exists. Skipping."
#    else
#        echo "--> '$dir' is missing. Starting transfer..."
#        # Transfer the directory using rsync
#        rsync -avz "$dir" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_TARGET}/"
#    fi
#
#    echo "------------------------------------------"
#done


#!/bin/bash

# Configuration
REMOTE_USER="cl1704"
REMOTE_HOST="140.180.240.66"
REMOTE_DIR="/Users/cl1704/PycharmProjects/embedding_dimension/BWM"
LOCAL_DIR="Data"

# Create local base directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

echo "Fetching list of subdirectories from remote..."

# Get the list of subdirectories (excluding the parent directory itself)
subdirs=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" "find ${REMOTE_DIR} -maxdepth 1 -type d -not -path ${REMOTE_DIR}")

for dir in $subdirs; do
    dir_name=$(basename "$dir")
    LOCAL_PATH="${LOCAL_DIR}/${dir_name}"

    echo "-----------------------------------------------"

    # CHECK: Does the directory already exist locally?
    if [ -d "$LOCAL_PATH" ]; then
        echo "Check: '$dir_name' already exists. Skipping download."
        continue
    fi

    echo "Downloading: $dir_name"

    # Perform the download
    rsync -avz "${REMOTE_USER}@${REMOTE_HOST}:${dir}/" "$LOCAL_PATH/"

    if [ $? -eq 0 ]; then
        echo "Successfully downloaded $dir_name"
    else
        echo "Error: Failed to download $dir_name"
    fi
done

echo "-----------------------------------------------"
echo "Process complete."