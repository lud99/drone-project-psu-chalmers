#!/bin/bash

# Start backend with no command, do allow debug
export backend_command="sleep infinity"

# Start everything else
docker compose up -d