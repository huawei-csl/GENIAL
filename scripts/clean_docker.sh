#!/bin/bash

set -x
set -e 
source .env

bash $SRC_DIR/scripts/close_all_docker_containers.sh
python $SRC_DIR/scripts/cleanup_all_volumes.py
