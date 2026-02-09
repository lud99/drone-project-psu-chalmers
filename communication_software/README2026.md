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

If not all appear, run ```docker compose restart``` until all of them are visible. Don't know why, but it worked for me after a few attempts.
You can start a single container with ```docker compose up <name>```. Note the lack of ```-d```. ```-d```stars in detatched mode, so you dont see any command output.
Not including it means you can see the logs.

Running ```docker compose up -d``` (without --build) will be substantially faster, any changes to the files on the host system will not be copied over (except those in a volume, wich the source code should be). You can also run ```docker compose restart <name>```.

To see the logs of a container after it has started, or is detatched, run ```docker compose logs -f <name>```
Access the terminal of the container: ```docker compose exec <name> bash```

If you have made a change to a dockerfile, you must run ```--build```. If you have made changes to sourcode, then you shouldn't have to.
Try to avoid running --build as it will build a new image, and after many runs your disk will be full

# How to debug and work with the containers

I have changed to we use a volume instead of copying the source files to the container. Practically, this means that any changes made to the container will actually persist. You can use the normal way of starting the docker containers, but you will not be able to debug any python code, nor easliy apply your changes without having to restart the container. It is your preference how you wish to do.

## Work and debug inside a container
The programming and debugging of the backend should be done in the docker container, for the respective submodule you wish to work on.

Install the Vs Code extensions ```Docker```and ```Dev Containers```

Run the script depending on what you want to debug, in a git bash terminal window: ```debug_all.sh, debug_backend.sh or begug_image_stitching.sh```. The one you aren't debugging will automatically start when the container starts, as usual. Note: These wont rebuild the dockerfile, if you have made any changes

Now in Vs Code press ```Ctrl + P, type >Attach to running container``` and select the python container you wish to debug (backend or image_stitching).
It will open a new window and load. Install the python (debugger) extension in your attatched Vs Code window, and wait. You will have to do this every time you attach.
https://marketplace.visualstudio.com/items?itemName=ms-python.python

Press ```F5``` to start a debugger, voila! you can now edit code and restart without having to restart the entire container.

If you restart the container(s), you will have to reload the vs code window and wait some time, aswell as install the extensions again

## Frontend
Rebuilding the frontend container depends on the backend starting. The webcontents are mounted as volume so that editing the files are just as simple as that.


### 5. Firewall and Network Configuration

*   **Firewall Interaction:**
    If you are using a host-based firewall on your Linux system, such as `ufw` (Uncomplicated Firewall) or `firewalld` (which may utilize `iptables-nft` or `iptables-legacy` as their backend), Docker is designed to manage its own network rules. This generally means that the ports exposed by the containers (as defined in your `docker-compose.yml` file) should automatically become accessible from other devices on the same local network (VLAN). Typically, you do not need to manually configure your host firewall for Docker's published ports, as Docker's rules take precedence.

    Issues may arise when connecting the android app to the backend, where the connection will suddenly be closed. We (think) we fixed it by disabling the windows firewall on the computer running the backend.

*   **Important Note for Eduroam (and similar isolated networks):**
    Networks like **Eduroam** (commonly found in academic institutions) often implement a security feature called "client isolation" or "AP isolation." This feature places each connected device into its own separate virtual LAN (VLAN), effectively preventing direct peer-to-peer communication between devices (e.g., your Linux machine running the backend and your Android phone) even if they are connected to the same Wi-Fi access point.

    **Workaround:** To enable communication between your backend service (running in Docker) and a client device (like an Android phone) when on such a network:
    1.  Connect both the Linux machine (hosting the Docker backend) and your client device to a network that **does not** enforce client isolation.
    2.  A common and effective solution is to use a **mobile hotspot** on your smartphone. Connect both devices to this hotspot.
    3.  Alternatively, using a **personal Wi-Fi router** at home will also typically allow inter-device communication on the local network.

### 6. GPU Acceleration

There is support for Nvidia GPU acceleration.
The library `nvidia-container-toolkit` has to be installed.
Use the `docker-compose-gpu.yml` file instead.
```bash
docker compose -f docker-compose-gpu.yml up -d
```
