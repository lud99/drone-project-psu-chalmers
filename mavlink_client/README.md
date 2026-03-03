# MAVLink client

Python client that **replicates the behaviour of the DJI Android app** but for MAVLink vehicles (e.g. ArduPilot, PX4). It talks to the **same backend** over the same WebSocket protocol, so the backend and frontend do not need to change. Everything is resolved automatically: no hardcoded IPs, ports, or radio connections.

---

## What this is

- **Purpose:** Run a MAVLink drone in the same system as the DJI Android app (same backend, same messages).
- **Protocol:** Same as Android: `drone_registration`, `telemetry`, `ping`, and commands (`flight_arm`, `flight_take_off`, `flight_return_to_home`, `task`, `abort_task`, `Coordinate_request`).
- **Automatic:** Backend URL from multicast discovery or last-used file; MAVLink connection auto-detected (UDP then serial); drone ID from vehicle heartbeat unless you set it.

---

## Setup

```bash
cd mavlink_client
pip install -r requirements.txt
```

Dependencies: `pymavlink`, `pydantic`, `websockets`.

---

## How to run

From repo root:

```bash
python mavlink_client/run.py
```

Or from inside `mavlink_client`:

```bash
python run.py
```

**Before running:** Backend must be running and advertising (multicast relay if in Docker). Vehicle or SITL must be on (e.g. sending to UDP 14550).

---

## What gets resolved automatically (no hardcoding)

1. **Backend WebSocket URL**  
   - If `BACKEND_WS_URL` is set → use it.  
   - Else listen for **multicast** on `239.255.42.99:9992` (same as Android) for up to 15 s.  
   - Else use **last used URL** from `~/.config/drone-project-psu-chalmers/mavlink_backend_url.txt`.  
   - After a successful connect, the URL is saved for next time.

2. **MAVLink connection**  
   - If `MAVLINK_CONNECTION` is set → use it.  
   - Else **auto-detect**: try `udp:0.0.0.0:14550`, then `udp:127.0.0.1:14550`, then serial (e.g. `/dev/ttyUSB0`, `/dev/ttyACM0` at 57600 on Linux; `/dev/tty.usbmodem*` on macOS). First link that receives a heartbeat is used.

3. **Drone ID**  
   - If `DRONE_ID` is set → use it.  
   - Else use `mavlink-{system_id}` from the vehicle’s first MAVLink heartbeat.

4. **Ports**  
   - Backend port comes from discovery or saved URL.  
   - MAVLink 14550 / 57600 are only tried during auto-detect, not hardcoded as the only option.

---

## Optional env overrides

| Variable | Effect |
|----------|--------|
| `BACKEND_WS_URL` | Force backend URL (e.g. `ws://192.168.1.100:14500`). |
| `MAVLINK_CONNECTION` | Force MAVLink connection (e.g. `udp:127.0.0.1:14550`, `serial:/dev/ttyUSB0:57600`). |
| `DRONE_ID` | Force drone ID (else from heartbeat). |
| `DRONE_MODEL` | Model name in registration (default `MAVLink drone`). |

---

## What’s implemented (Android checklist)

Everything the Android app does for the backend is covered, except local camera/WebRTC video (we send “no camera” and ignore WebRTC signaling).

| # | Android behaviour | MAVLink client |
|---|-------------------|----------------|
| **A** | DJI SDK registration + product connection | Connect to vehicle (UDP/serial). No cloud registration. Identify by MAV system ID. |
| **B** | Connection lifecycle, UI updates | Connected/disconnected state; optional callbacks. |
| **C** | Backend WebSocket + discovery | Same: manual URL (env), multicast discovery (239.255.42.99:9992), last-used persisted. Same WebSocket URL format. |
| **D** | Telemetry to backend (1 Hz) | Same: `msg_type: "telemetry"` with `lat`, `lon`, `alt`, `heading`, `speed`, `battery_percent` from GLOBAL_POSITION_INT, VFR_HUD, BATTERY_STATUS. |
| **E** | Drone registration (capabilities) | One-time `drone_registration` with `drone_type: "MAVLink"`, model, drone_id, capabilities (camera/led/spotlight/speaker/max_speed). |
| **F** | Heartbeat to backend (5 s) | Same: `{"msg_type": "ping"}` every 5 s. |
| **G** | Backend commands | `Coordinate_request` → reply with current position. `flight_arm` → arm. `flight_take_off` → takeoff. `flight_return_to_home` → RTL. `task` (go_to, go_home, land) → execute and send `task_event`. `abort_task` (go_home, hover, land) → RTL/land/hover. `offer`/`candidate`/`answer` (WebRTC) → ignored (stub). |
| **H** | Mission / waypoints | Arm, takeoff, RTL, land, single waypoint (go_to). |
| **I** | Camera feed (local) | Not implemented; capabilities.camera = null. |
| **J** | Gimbal control | Not implemented (no gimbal over MAVLink in this client). |
| **K** | WebRTC video to backend | Not implemented; WebRTC messages ignored. |
| **L** | DroneAdapter / task protocol | Task execution and `task_event` (task_complete / task_failed); abort_task with go_home/land/hover. |
| **M** | Coordinates from backend | On `Coordinate_request`, reply with current position (lat, lng, alt, angle). |
| **N** | Debug message | Backend can send; client does not send debug unless extended. |

---

## How it works (flow)

1. **Resolve backend URL** (env → multicast → persisted). Exit with a clear message if none found.
2. **Resolve MAVLink connection** (env → auto-detect UDP/serial). Exit if no heartbeat.
3. **Connect to vehicle**, start receive thread (telemetry state updated from GLOBAL_POSITION_INT, VFR_HUD, BATTERY_STATUS, HEARTBEAT).
4. **Wait for first heartbeat** to get system_id; then set drone_id = env or `mavlink-{system_id}`.
5. **Connect to backend WebSocket**, send `drone_registration` once, then loop: send **telemetry** every 1 s, **ping** every 5 s, and **handle incoming messages** (flight_arm, flight_take_off, flight_return_to_home, task, abort_task, Coordinate_request, WebRTC stub).
6. **Persist backend URL** on successful connect for next run.

---

## If something fails

- **“Backend URL not found”**  
  Start the backend and ensure it advertises (multicast relay if in Docker). Or set `BACKEND_WS_URL=ws://host:port`.

- **“No MAVLink vehicle found”**  
  Start SITL or connect the vehicle. Ensure something is sending on UDP 14550 or the serial port. Or set `MAVLINK_CONNECTION=udp:127.0.0.1:14550` (or your serial path).

---

## File layout

| File | Role |
|------|------|
| `run.py` | Entrypoint: resolve backend + connection, connect vehicle + backend, run loops. |
| `requirements.txt` | pymavlink, pydantic, websockets. |
| `mavlink_client/__init__.py` | Package init. |
| `mavlink_client/config.py` | Config dataclasses and resolvers: `resolve_backend_url()`, `resolve_connection()`, `resolve_drone_id()`. No hardcoded URLs or connection strings. |
| `mavlink_client/discovery.py` | Multicast listener (239.255.42.99:9992), parse CTH+JSON; `persist_backend_url()`, `load_persisted_backend_url()`. |
| `mavlink_client/connection_auto.py` | Auto-detect MAVLink: try UDP then serial until one gets a heartbeat. |
| `mavlink_client/messages.py` | Wire-format schemas (TelemetryMessage, DroneRegistrationMessage, TaskMessage, AbortTaskMessage, task_event, etc.) aligned with backend. |
| `mavlink_client/mavlink_connection.py` | MAVLink connection, receive thread, telemetry state (lat, lon, alt, heading, speed, battery_percent, system_id from heartbeat). |
| `mavlink_client/mission_handler.py` | Send commands: arm, disarm, takeoff, RTL, land, go_to (COMMAND_LONG / COMMAND_INT). |
| `mavlink_client/backend_client.py` | WebSocket client: send registration, telemetry, ping; handle commands; persist URL on connect; WebRTC stub. |

All of the above are updated and consistent with the behaviour described in this README.
