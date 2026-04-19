#!/bin/bash

set -e

python3 export_host_ip.py
docker compose up backend -d --build