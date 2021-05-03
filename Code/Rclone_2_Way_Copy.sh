#!/bin/bash

#SBATCH --account=pi-cpatt
#SBATCH --partition=standard
#SBATCH --job-name=Rclone_2_Way_Copy

# Suppress output
#SBATCH --output=/dev/null

# Push to remote/online Dropbox
rclone copy ~/Box/ECMA-31330-Project Box:ECMA-31330-Project --update
# Pull from remote/online Dropbox
rclone copy Box:ECMA-31330-Project ~/Box/ECMA-31330-Project --update
