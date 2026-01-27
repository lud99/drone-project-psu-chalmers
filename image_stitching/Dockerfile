# FROM python:3.13-slim
FROM nvidia/cuda:12.8.1-base-ubuntu24.04
# Match the cuda version with $ nvcc --version
# See available images here: https://hub.docker.com/r/nvidia/cuda/tags

ENV WORKSPACE_DIR=/root/image_stitching
RUN mkdir -p ${WORKSPACE_DIR}

COPY requirements.txt ${WORKSPACE_DIR}


RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    #libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages -r ${WORKSPACE_DIR}/requirements.txt

RUN apt-get update && apt-get install -y libgl1 libglx-mesa0

# vscode is so weird and refused to open WORKSPACE_DIR, and instead uses /root. so have to code the .vscode older to there
COPY .vscode root/.vscode

WORKDIR ${WORKSPACE_DIR}
CMD ["/bin/bash"]
# COPY yolov8s.pt .

# CMD ["python3", "main.py"]