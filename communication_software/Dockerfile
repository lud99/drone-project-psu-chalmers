ARG ROS_DISTRO=humble
ARG OS=jammy
ARG FROM_IMAGE=astazero/atos_docker_env:latest

FROM ${FROM_IMAGE} AS application

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]


RUN pip install redis aiortc geopy debugpy

# --- Workspace Setup ---
ENV WORKSPACE_DIR=/root/atos_ws
RUN mkdir -p ${WORKSPACE_DIR}/src

# Copy the entire project into the workspace's src directory
COPY . ${WORKSPACE_DIR}/src/communication_software
COPY .vscode ${WORKSPACE_DIR}/.vscode

EXPOSE 5000  

# --- Build ---
WORKDIR ${WORKSPACE_DIR}

# Build the ROS 2 package (using bash -c for robustness)
RUN bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build --symlink-install"

# Add sourcing of the workspace setup to .bashrc (for interactive shells)
RUN echo "source ${WORKSPACE_DIR}/install/local_setup.bash" >> ~/.bashrc
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc


# Default command (can be overridden in docker-compose.yml)
CMD ["/bin/bash"]
# CMD ["ros2", "run", "communication_software", "run_main"]
