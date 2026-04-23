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
background_tasks: Dict[str, Any] = {}


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
    # Spawn background task for takeoff (long-running operation)
    cmd = Command(type=CommandType.TAKEOFF_TO_RELATIVE_ALTITUDE, data=data)
    task = asyncio.create_task(controller.execute_command(cmd))
    background_tasks['takeoff'] = {'task': task, 'state': 'running'}
    return {"success": True, "message": "Takeoff command accepted, executing in background", "task_id": "takeoff"}


@app.post("/api/land")
async def land():
    if not controller:
        raise HTTPException(
            status_code=503, detail="Controller not initialized")
    # Spawn background task for land (long-running operation)
    cmd = Command(type=CommandType.LAND)
    task = asyncio.create_task(controller.execute_command(cmd))
    background_tasks['land'] = {'task': task, 'state': 'running'}
    return {"success": True, "message": "Land command accepted, executing in background", "task_id": "land"}


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


@app.get("/api/disarm")
async def disarm_get():
    # Browser address bar requests are GET; keep behavior aligned with POST.
    return await disarm()


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


@app.post("/api/geofence/circle")
async def set_circular_geofence(data: Dict[str, Any]):
    if not controller:
        raise HTTPException(
            status_code=503, detail="Controller not initialized")
    cmd = Command(type=CommandType.SET_CIRCULAR_GEOFENCE, data=data)
    result = await controller.execute_command(cmd)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.delete("/api/geofence/circle")
async def clear_circular_geofence():
    if not controller:
        raise HTTPException(
            status_code=503, detail="Controller not initialized")
    cmd = Command(type=CommandType.CLEAR_CIRCULAR_GEOFENCE)
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

    # Include background task status
    task_status = {}
    for task_id, task_info in list(background_tasks.items()):
        task = task_info.get('task')
        if task and not task.done():
            task_status[task_id] = 'running'
        elif task and task.done():
            try:
                result_data = task.result()
                task_status[task_id] = result_data.get('message', 'completed')
                # Clean up finished task
                del background_tasks[task_id]
            except Exception as e:
                task_status[task_id] = f'error: {str(e)}'
                del background_tasks[task_id]

    result['background_tasks'] = task_status
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
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { display: flex; gap: 20px; }
        .panel { flex: 1; border: 1px solid #ccc; padding: 10px; background: #fff; border-radius: 6px; }
        .status { background: #f0f0f0; padding: 10px; margin-bottom: 10px; border-radius: 4px; font-size: 0.9em; }
        button { margin: 3px; padding: 8px 12px; cursor: pointer; border-radius: 4px; border: 1px solid #bbb; }
        input, textarea, select { margin: 3px; padding: 5px; border-radius: 4px; border: 1px solid #bbb; }
        #map { height: 420px; border-radius: 4px; }
        .fence-section { border: 1px solid #ccc; border-radius: 4px; padding: 10px; margin-top: 10px; background: #fafafa; }
        .fence-section h4 { margin: 0 0 4px 0; font-size: 0.95em; }
        .fence-section hr { border: none; border-top: 1px solid #ddd; margin: 4px 0 8px 0; }
        .fence-panel { display: none; border: 1px solid #ddd; padding: 8px; border-radius: 4px; margin-top: 6px; background: #fff; }
        .fence-panel p { font-size: 0.82em; color: #555; margin: 4px 0 8px 0; }
        .btn-apply  { background: #3a8; color: #fff; border-color: #2a7; }
        .btn-clear  { background: #c44; color: #fff; border-color: #b33; }
        .btn-active { background: #3a8; color: #fff; border-color: #2a7; }
        .fence-status { font-size: 0.83em; color: #555; min-height: 18px; }
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
            <input type="number" id="gotoAlt" placeholder="Rel Alt (m)" value="10">
            <div style="font-size:0.85em;color:#555;margin:10px 0 4px 0;">Relative target</div>
            <input type="number" id="gotoDistance" placeholder="Distance (m)" value="5" min="0.1" step="0.1">
            <select id="gotoDirection">
                <option value="N">North</option>
                <option value="S">South</option>
                <option value="E">East</option>
                <option value="W">West</option>
            </select>
            <button onclick="gotoRelative()">Go Relative</button>

            <h3>GeoFence</h3>
            <p style="font-size:0.85em;color:#555;margin:4px 0 8px 0;">
                Set a keep-in boundary. The drone will land or hold if it exits.
            </p>
            <button id="btnPolygonMode" onclick="setFenceMode('polygon')">&#9999; Polygon Fence</button>
            <button id="btnCircleMode"  onclick="setFenceMode('circle')" >&#8857; Circular Fence</button>

            <div id="polygonFencePanel" class="fence-panel">
                <b>Polygon Fence</b>
                <p>Click the map to add vertices (min 3). A dashed preview appears as you click.</p>
                <div id="polyPointsList" class="fence-status">No points added yet.</div>
                <div style="margin-top:6px;">
                    <button onclick="undoPolyPoint()">&#8617; Undo</button>
                    <button class="btn-apply" onclick="applyPolygonFence()">&#10004; Apply</button>
                    <button class="btn-clear" onclick="clearPolygonFence()">&#10006; Clear</button>
                </div>
            </div>

            <div id="circleFencePanel" class="fence-panel">
                <b>Circular Fence</b>
                <p>Click the map to place the center, then set the radius and apply.</p>
                <div>Center: <span id="circleCenterDisplay" class="fence-status">Not set &mdash; click the map</span></div>
                <div style="margin-top:6px;">
                    Radius: <input type="number" id="circleRadius" value="100" min="1" style="width:70px;"> m
                </div>
                <div style="margin-top:6px;">
                    <button class="btn-apply" onclick="applyCircularFence()">&#10004; Apply</button>
                    <button class="btn-clear" onclick="clearCircularFence()">&#10006; Clear</button>
                </div>
            </div>

            <div class="fence-section" style="margin-top:10px;">
                <h4>Polygon Fences</h4><hr>
                <div id="polygonFenceStatus" class="fence-status">None</div>
            </div>
            <div class="fence-section">
                <h4>Circular Fences</h4><hr>
                <div id="circularFenceStatus" class="fence-status">None</div>
            </div>
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
        let droneMarker   = null;
        let polygonLayer  = null;
        let circleLayer   = null;
        let fenceMode     = null;
        let polyDraft     = [];
        let draftPolyLayer    = null;
        let circleCenter      = null;
        let draftCenterMarker = null;

        function initMap() {
            map = L.map('map').setView([37.7749, -122.4194], 13);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; OpenStreetMap contributors'
            }).addTo(map);

            map.on('click', function(e) {
                if (fenceMode === 'polygon') {
                    polyDraft.push([e.latlng.lat, e.latlng.lng]);
                    updateDraftPoly();
                } else if (fenceMode === 'circle') {
                    circleCenter = [e.latlng.lat, e.latlng.lng];
                    document.getElementById('circleCenterDisplay').textContent =
                        circleCenter[0].toFixed(6) + ', ' + circleCenter[1].toFixed(6);
                    if (draftCenterMarker) map.removeLayer(draftCenterMarker);
                    draftCenterMarker = L.circleMarker(circleCenter, {radius: 7, color: '#e80'}).addTo(map);
                }
            });
        }

        function updateDraftPoly() {
            if (draftPolyLayer) { map.removeLayer(draftPolyLayer); draftPolyLayer = null; }
            const label = polyDraft.map((p, i) =>
                (i + 1) + ':(' + p[0].toFixed(5) + ',' + p[1].toFixed(5) + ')').join('  ');
            document.getElementById('polyPointsList').textContent =
                polyDraft.length ? label : 'No points added yet.';
            if (polyDraft.length >= 2) {
                draftPolyLayer = L.polyline(polyDraft, {color: '#e80', dashArray: '6,5'}).addTo(map);
            }
        }

        function undoPolyPoint() {
            polyDraft.pop();
            updateDraftPoly();
        }

        function setFenceMode(mode) {
            fenceMode = mode;
            document.getElementById('polygonFencePanel').style.display = mode === 'polygon' ? 'block' : 'none';
            document.getElementById('circleFencePanel').style.display  = mode === 'circle'  ? 'block' : 'none';
            document.getElementById('btnPolygonMode').className = mode === 'polygon' ? 'btn-active' : '';
            document.getElementById('btnCircleMode').className  = mode === 'circle'  ? 'btn-active' : '';
            if (mode === 'polygon') { polyDraft = []; updateDraftPoly(); }
            if (mode === 'circle') {
                circleCenter = null;
                document.getElementById('circleCenterDisplay').textContent = 'Not set \u2014 click the map';
                if (draftCenterMarker) { map.removeLayer(draftCenterMarker); draftCenterMarker = null; }
            }
        }

        async function applyPolygonFence() {
            if (polyDraft.length < 3) { alert('Need at least 3 points.'); return; }
            const polygon = polyDraft.map(p => ({latitude: p[0], longitude: p[1]}));
            const result = await apiCall('/api/geofence', 'POST', {polygon});
            if (result.detail) { alert('Error: ' + result.detail); return; }
            if (draftPolyLayer) { map.removeLayer(draftPolyLayer); draftPolyLayer = null; }
            polyDraft = []; updateDraftPoly();
            await updateStatus();
        }

        async function clearPolygonFence() {
            await apiCall('/api/geofence', 'DELETE');
            if (draftPolyLayer) { map.removeLayer(draftPolyLayer); draftPolyLayer = null; }
            polyDraft = []; updateDraftPoly();
            await updateStatus();
        }

        async function applyCircularFence() {
            if (!circleCenter) { alert('Click the map to set the center first.'); return; }
            const radius = parseFloat(document.getElementById('circleRadius').value);
            if (!radius || radius <= 0) { alert('Enter a valid radius greater than 0.'); return; }
            const result = await apiCall('/api/geofence/circle', 'POST', {
                latitude: circleCenter[0], longitude: circleCenter[1], radius_m: radius
            });
            if (result.detail) { alert('Error: ' + result.detail); return; }
            if (draftCenterMarker) { map.removeLayer(draftCenterMarker); draftCenterMarker = null; }
            circleCenter = null;
            document.getElementById('circleCenterDisplay').textContent = 'Not set \u2014 click the map';
            await updateStatus();
        }

        async function clearCircularFence() {
            await apiCall('/api/geofence/circle', 'DELETE');
            if (draftCenterMarker) { map.removeLayer(draftCenterMarker); draftCenterMarker = null; }
            circleCenter = null;
            document.getElementById('circleCenterDisplay').textContent = 'Not set \u2014 click the map';
            await updateStatus();
        }

        async function apiCall(endpoint, method='POST', data=null) {
            const options = { method };
            if (data) options.headers = {'Content-Type': 'application/json'};
            if (data) options.body = JSON.stringify(data);
            const response = await fetch(endpoint, options);
            const payload = await response.json();
            if (!response.ok) {
                const msg = payload.detail || payload.message || ('HTTP ' + response.status);
                throw new Error(msg);
            }
            return payload;
        }

        async function arm() {
            try { await apiCall('/api/arm'); await updateStatus(); }
            catch (e) { alert('Arm failed: ' + e.message); }
        }

        async function disarm() {
            try { await apiCall('/api/disarm'); await updateStatus(); }
            catch (e) { alert('Disarm failed: ' + e.message); }
        }

        async function takeoff() {
            const alt = parseFloat(document.getElementById('takeoffAlt').value);
            try {
                await apiCall('/api/takeoff', 'POST', {relative_altitude_m: alt});
                await updateStatus();
            } catch (e) {
                alert('Takeoff failed: ' + e.message);
            }
        }

        async function land() {
            try { await apiCall('/api/land'); await updateStatus(); }
            catch (e) { alert('Land failed: ' + e.message); }
        }

        async function hold() {
            try { await apiCall('/api/hold'); await updateStatus(); }
            catch (e) { alert('Hold failed: ' + e.message); }
        }

        async function getCurrentState() {
            const status = await apiCall('/api/status', 'GET');
            return (status.state || '').toLowerCase();
        }

        async function getCurrentStatus() {
            return await apiCall('/api/status', 'GET');
        }

        async function gotoRelative() {
            const distance = parseFloat(document.getElementById('gotoDistance').value);
            const direction = document.getElementById('gotoDirection').value;
            const alt = parseFloat(document.getElementById('gotoAlt').value);
            try {
                if (!Number.isFinite(distance) || distance <= 0) {
                    alert('Relative goto requires a valid distance greater than 0.');
                    return;
                }
                if (!Number.isFinite(alt) || alt <= 0) {
                    alert('Relative goto requires a valid relative altitude greater than 0.');
                    return;
                }

                const status = await getCurrentStatus();
                const state = (status.state || '').toLowerCase();
                if (state !== 'hovering' && state !== 'navigating' && state !== 'holding') {
                    alert('Relative goto requires the drone to be airborne. Take off first and wait for hovering state.');
                    return;
                }

                const telem = status.telemetry || {};
                if (telem.armed !== true) {
                    alert('Relative goto requires the drone to be armed. Arm and take off first.');
                    return;
                }
                if ((telem.gps_fix_type ?? 0) < 3) {
                    alert('Relative goto requires GPS fix type 3 or better.');
                    return;
                }

                await apiCall('/api/goto', 'POST', {
                    distance_m: distance,
                    direction: direction,
                    relative_altitude_m: alt
                });
                await updateStatus();
            } catch (e) {
                alert('Relative goto failed: ' + e.message);
            }
        }

        async function updateStatus() {
            const status = await apiCall('/api/status', 'GET');
            const polyInfo = status.polygon
                ? 'Active (' + (status.polygon.length - 1) + ' vertices)'
                : 'None';
            const circInfo = status.circle
                ? 'Active (r\u202f=\u202f' + status.circle.radius_m + '\u202fm)'
                : 'None';
            
            // Build task status string
            let taskStatusStr = '';
            if (status.background_tasks && Object.keys(status.background_tasks).length > 0) {
                taskStatusStr = '<br><strong style="color: #f80;">Background Tasks:</strong> ';
                for (const [taskId, taskMsg] of Object.entries(status.background_tasks)) {
                    taskStatusStr += `<br>&nbsp;&nbsp;${taskId}: ${taskMsg}`;
                }
            }
            
            document.getElementById('status').innerHTML =
                'State: '    + status.state + '<br>' +
                'Command: '  + (status.current_command || 'None') + '<br>' +
                'Polygon fence: ' + polyInfo + '<br>' +
                'Circular fence: ' + circInfo + '<br>' +
                'Safety: '   + (status.latest_safety_message || 'None') + '<br>' +
                'Backend: '  + (status.using_mock_drone ? 'Mock' : 'Real MAVLink') +
                taskStatusStr;

            const telem = status.telemetry;
            document.getElementById('telemetry').innerHTML =
                'Lat: ' + telem.lat + '<br>' +
                'Lon: ' + telem.lon + '<br>' +
                'Alt: ' + telem.alt + '<br>' +
                'Heading: ' + telem.heading + '<br>' +
                'Speed: ' + telem.speed + '<br>' +
                'Battery: ' + telem.battery_percent + '%<br>' +
                'Mode: ' + telem.mode + '<br>' +
                'Armed: ' + telem.armed + '<br>' +
                'GPS Fix: ' + telem.gps_fix_type + '<br>' +
                'Sats: ' + telem.satellites_visible;

            // Drone marker
            if (telem.lat && telem.lon) {
                const pos = [telem.lat, telem.lon];
                if (!droneMarker) droneMarker = L.marker(pos).addTo(map);
                else droneMarker.setLatLng(pos);
                map.setView(pos);
            }

            // Polygon fence layer
            if (status.polygon) {
                const coords = status.polygon.map(p => [p.latitude, p.longitude]);
                if (!polygonLayer) polygonLayer = L.polygon(coords, {color: '#e64', fillOpacity: 0.12}).addTo(map);
                else polygonLayer.setLatLngs(coords);
                document.getElementById('polygonFenceStatus').textContent =
                    (status.polygon.length - 1) + ' vertices';
            } else {
                if (polygonLayer) { map.removeLayer(polygonLayer); polygonLayer = null; }
                document.getElementById('polygonFenceStatus').textContent = 'None';
            }

            // Circular fence layer
            if (status.circle) {
                const c = status.circle;
                if (!circleLayer) {
                    circleLayer = L.circle([c.latitude, c.longitude], {
                        radius: c.radius_m, color: '#46e', fillOpacity: 0.1
                    }).addTo(map);
                } else {
                    circleLayer.setLatLng([c.latitude, c.longitude]);
                    circleLayer.setRadius(c.radius_m);
                }
                document.getElementById('circularFenceStatus').textContent =
                    'Center: ' + c.latitude.toFixed(6) + ', ' + c.longitude.toFixed(6) +
                    ' \u2014 Radius: ' + c.radius_m + ' m';
            } else {
                if (circleLayer) { map.removeLayer(circleLayer); circleLayer = null; }
                document.getElementById('circularFenceStatus').textContent = 'None';
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
