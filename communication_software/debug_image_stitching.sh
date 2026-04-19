#!/bin/bash

set -e

python3 export_host_ip.py

# Start backend with no command, do allow debug
export image_stitching_command="sleep infinity"

# Start everything else
docker compose up -d