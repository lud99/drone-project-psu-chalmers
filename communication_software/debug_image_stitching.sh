#!/bin/bash

# Start backend with no command, do allow debug
export image_stitching_command="sleep infinity"

# Start everything else
docker compose up -d --build