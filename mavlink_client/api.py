from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio
from typing import Dict, Any, Optional
import logging

from .config import Config
from .drone_controller import DroneController
from .commands import Command, CommandType

logger = logging.getLogger(__name__)

app = FastAPI(title="Drone Control API", version="1.0.0")

controller: Optional[DroneController] = None


@app.post("/api/arm")
async def arm():
    if not controller:
        raise HTTPException(
            status_code=503, detail="Controller not initialized")
    cmd = Command(type=CommandType.ARM)
    result = await controller.execute_command(cmd)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/api/takeoff")
async def takeoff(data: Dict[str, float]):
    if not controller:
        raise HTTPException(
            status_code=503, detail="Controller not initialized")
    cmd = Command(type=CommandType.TAKEOFF_TO_RELATIVE_ALTITUDE, data=data)
    result = await controller.execute_command(cmd)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/api/land")
async def land():
    if not controller:
        raise HTTPException(
            status_code=503, detail="Controller not initialized")
    cmd = Command(type=CommandType.LAND)
    result = await controller.execute_command(cmd)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/api/disarm")
async def disarm():
    if not controller:
        raise HTTPException(
            status_code=503, detail="Controller not initialized")
    cmd = Command(type=CommandType.DISARM)
    result = await controller.execute_command(cmd)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/api/goto")
async def goto(data: Dict[str, Any]):
    if not controller:
        raise HTTPException(
            status_code=503, detail="Controller not initialized")
    cmd = Command(type=CommandType.GOTO_POINT, data=data)
    result = await controller.execute_command(cmd)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/api/geofence")
async def set_geofence(data: Dict[str, Any]):
    if not controller:
        raise HTTPException(
            status_code=503, detail="Controller not initialized")
    cmd = Command(type=CommandType.SET_POLYGON_GEOFENCE, data=data)
    result = await controller.execute_command(cmd)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.delete("/api/geofence")
async def clear_geofence():
    if not controller:
        raise HTTPException(
            status_code=503, detail="Controller not initialized")
    cmd = Command(type=CommandType.CLEAR_POLYGON_GEOFENCE)
    result = await controller.execute_command(cmd)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/api/hold")
async def hold():
    if not controller:
        raise HTTPException(
            status_code=503, detail="Controller not initialized")
    cmd = Command(type=CommandType.HOLD)
    result = await controller.execute_command(cmd)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.get("/api/status")
async def get_status():
    if not controller:
        raise HTTPException(
            status_code=503, detail="Controller not initialized")
    cmd = Command(type=CommandType.GET_STATUS)
    result = await controller.execute_command(cmd)
    return result


@app.get("/api/telemetry")
async def get_telemetry():
    if not controller:
        raise HTTPException(
            status_code=503, detail="Controller not initialized")
    return await controller._get_status()


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Drone Control Interface</title>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { display: flex; gap: 20px; }
        .panel { flex: 1; border: 1px solid #ccc; padding: 10px; }
        .status { background: #f0f0f0; padding: 10px; margin-bottom: 10px; }
        button { margin: 5px; padding: 10px; }
        input { margin: 5px; padding: 5px; }
        #map { height: 400px; }
    </style>
</head>
<body>
    <h1>Drone Control Interface</h1>
    <div class="container">
        <div class="panel">
            <h2>Commands</h2>
            <button onclick="arm()">Arm</button>
            <button onclick="disarm()">Disarm</button>
            <br>
            <input type="number" id="takeoffAlt" placeholder="Relative Alt (m)" value="2">
            <button onclick="takeoff()">Takeoff</button>
            <br>
            <button onclick="land()">Land</button>
            <button onclick="hold()">Hold</button>
            <br>
            <h3>Goto</h3>
            <input type="number" id="gotoLat" placeholder="Latitude" step="0.000001">
            <input type="number" id="gotoLon" placeholder="Longitude" step="0.000001">
            <input type="number" id="gotoAlt" placeholder="Rel Alt (m)" value="10">
            <button onclick="gotoPoint()">Go To</button>
            <br>
            <h3>Geofence</h3>
            <textarea id="polygonJson" placeholder='[{"latitude": 37.7749, "longitude": -122.4194}, ...]' rows="4"></textarea>
            <br>
            <button onclick="setGeofence()">Set Geofence</button>
            <button onclick="clearGeofence()">Clear Geofence</button>
        </div>
        <div class="panel">
            <h2>Status</h2>
            <div id="status" class="status">Loading...</div>
            <h2>Telemetry</h2>
            <div id="telemetry" class="status">Loading...</div>
            <h2>Map</h2>
            <div id="map"></div>
        </div>
    </div>
    <script>
        let map;
        let droneMarker;
        let polygonLayer;

        function initMap() {
            map = L.map('map').setView([37.7749, -122.4194], 13);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            }).addTo(map);
        }

        async function apiCall(endpoint, method='POST', data=null) {
            const options = { method };
            if (data) options.headers = {'Content-Type': 'application/json'};
            if (data) options.body = JSON.stringify(data);
            const response = await fetch(endpoint, options);
            return await response.json();
        }

        async function arm() { await apiCall('/api/arm'); updateStatus(); }
        async function disarm() { await apiCall('/api/disarm'); updateStatus(); }
        async function takeoff() {
            const alt = parseFloat(document.getElementById('takeoffAlt').value);
            await apiCall('/api/takeoff', 'POST', {relative_altitude_m: alt});
            updateStatus();
        }
        async function land() { await apiCall('/api/land'); updateStatus(); }
        async function hold() { await apiCall('/api/hold'); updateStatus(); }
        async function gotoPoint() {
            const lat = parseFloat(document.getElementById('gotoLat').value);
            const lon = parseFloat(document.getElementById('gotoLon').value);
            const alt = parseFloat(document.getElementById('gotoAlt').value);
            await apiCall('/api/goto', 'POST', {latitude: lat, longitude: lon, relative_altitude_m: alt});
            updateStatus();
        }
        async function setGeofence() {
            try {
                const polygon = JSON.parse(document.getElementById('polygonJson').value);
                await apiCall('/api/geofence', 'POST', {polygon});
                updateStatus();
            } catch (e) {
                alert('Invalid JSON');
            }
        }
        async function clearGeofence() { await apiCall('/api/geofence', 'DELETE'); updateStatus(); }

        async function updateStatus() {
            const status = await apiCall('/api/status', 'GET');
            document.getElementById('status').innerHTML = `
                State: ${status.state}<br>
                Command: ${status.current_command || 'None'}<br>
                Geofence: ${status.geofence_active ? 'Active' : 'Inactive'}<br>
                Safety: ${status.latest_safety_message || 'None'}<br>
                Backend: ${status.using_mock_drone ? 'Mock' : 'Real MAVLink'}
            `;
            const telem = status.telemetry;
            document.getElementById('telemetry').innerHTML = `
                Lat: ${telem.lat}<br>
                Lon: ${telem.lon}<br>
                Alt: ${telem.alt}<br>
                Heading: ${telem.heading}<br>
                Speed: ${telem.speed}<br>
                Battery: ${telem.battery_percent}%<br>
                Mode: ${telem.mode}<br>
                Armed: ${telem.armed}<br>
                GPS Fix: ${telem.gps_fix_type}<br>
                Sats: ${telem.satellites_visible}
            `;
            // Update map
            if (telem.lat && telem.lon) {
                const pos = [telem.lat, telem.lon];
                if (!droneMarker) {
                    droneMarker = L.marker(pos).addTo(map);
                } else {
                    droneMarker.setLatLng(pos);
                }
                map.setView(pos);
            }
            if (status.polygon && !polygonLayer) {
                const coords = status.polygon.map(p => [p.latitude, p.longitude]);
                polygonLayer = L.polygon(coords).addTo(map);
            } else if (!status.polygon && polygonLayer) {
                map.removeLayer(polygonLayer);
                polygonLayer = null;
            }
        }

        initMap();
        updateStatus();
        setInterval(updateStatus, 1000);
    </script>
</body>
</html>
    """


@app.get("/favicon.ico")
async def favicon() -> Response:
    # Prevent repeated browser 404s when no favicon file is provided.
    return Response(status_code=204)


@app.get("/apple-touch-icon.png")
async def apple_touch_icon() -> Response:
    return Response(status_code=204)


@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools() -> Response:
    return Response(status_code=204)


def start_api_server(ctrl: DroneController, config: Config):
    global controller
    controller = ctrl
    uvicorn.run(app, host="0.0.0.0", port=config.api_port)
