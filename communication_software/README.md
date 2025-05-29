# Communication Software - Backend Setup (2025 Version)

This guide outlines the steps to set up the 2025 version of the Communication Software backend using Docker on a Linux system.

## 1. Prerequisites

Before you begin, ensure you have the following installed on your Linux system:

1.  **Git:** To clone the repository. (e.g., `sudo apt update && sudo apt install git`)
2.  **Docker Engine:** Follow the official installation guide for your Linux distribution: [Install Docker Engine](https://docs.docker.com/engine/install/)
3.  **Docker Compose:** This is usually included with Docker Desktop or can be installed as a plugin for Docker Engine. Verify by running `docker compose version`. If not found, see: [Install Docker Compose](https://docs.docker.com/compose/install/)

### 2. Navigate to the Project Directory

Ensure your terminal is in the `communication_software` directory, which is located at the same level as this `README.md` file (if this `README.md` is in `$GIT_ROOT_REPOSITORY/communication_software/`). This directory contains the `docker-compose.yml` file required for the next steps.

If your current directory is the root of the cloned repository (`$GIT_ROOT_REPOSITORY`), use:

```bash
cd communication_software
```

### 3. Build and Start the Services
To build the Docker images and start all the application services in detached mode (running in the background), execute the following command:
```bash
docker compose up -d --build
```
**Important Note on Build Time:**
The very first time you run this command, the build process might take a considerable amount of time (approximately 700 seconds or more). Subsequent builds will be significantly faster because Docker caches the image layers.

### 4. Verify Running Services
Once the `docker compose up` command has finished, you can verify that all services are running as expected:
```bash
docker compose ps
```

You should see an output listing the following containers, typically with a status indicating they are "Up" or "Running":
- astazero-redis
- atos
- isoObject
- image_stitching
- backend
- frontend

### 5. Firewall and Network Configuration

*   **Firewall Interaction:**
    If you are using a host-based firewall on your Linux system, such as `ufw` (Uncomplicated Firewall) or `firewalld` (which may utilize `iptables-nft` or `iptables-legacy` as their backend), Docker is designed to manage its own network rules. This generally means that the ports exposed by the containers (as defined in your `docker-compose.yml` file) should automatically become accessible from other devices on the same local network (VLAN). Typically, you do not need to manually configure your host firewall for Docker's published ports, as Docker's rules take precedence.

*   **Important Note for Eduroam (and similar isolated networks):**
    Networks like **Eduroam** (commonly found in academic institutions) often implement a security feature called "client isolation" or "AP isolation." This feature places each connected device into its own separate virtual LAN (VLAN), effectively preventing direct peer-to-peer communication between devices (e.g., your Linux machine running the backend and your Android phone) even if they are connected to the same Wi-Fi access point.

    **Workaround:** To enable communication between your backend service (running in Docker) and a client device (like an Android phone) when on such a network:
    1.  Connect both the Linux machine (hosting the Docker backend) and your client device to a network that **does not** enforce client isolation.
    2.  A common and effective solution is to use a **mobile hotspot** on your smartphone. Connect both devices to this hotspot.
    3.  Alternatively, using a **personal Wi-Fi router** at home will also typically allow inter-device communication on the local network.
