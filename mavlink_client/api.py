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
    <script src="https://cdn.jsdelivr.net/npm/hls.js@1"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Inter', Arial, sans-serif; background: #f0f4f8; color: #0f172a; min-height: 100vh; }

        /* TOP BAR */
        .top-bar { display: flex; align-items: center; gap: 12px; padding: 9px 16px; background: #ffffff; border-bottom: 0.5px solid #e2e8f0; position: sticky; top: 0; z-index: 2000; }
        .logo-dot { width: 8px; height: 8px; border-radius: 50%; background: #22d3ee; box-shadow: 0 0 7px #22d3ee55; flex-shrink: 0; }
        .app-title { font-size: 10px; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: #64748b; }
        .badge { font-size: 9px; letter-spacing: 0.05em; text-transform: uppercase; color: #64748b; background: #f1f5f9; border: 0.5px solid #e2e8f0; border-radius: 99px; padding: 2px 8px; }
        .ci-list { display: flex; gap: 14px; margin-left: auto; }
        .ci-item { display: flex; align-items: center; gap: 5px; font-size: 9px; text-transform: uppercase; letter-spacing: 0.06em; color: #64748b; }
        .ci-dot { width: 6px; height: 6px; border-radius: 50%; background: #cbd5e1; transition: background 0.3s; }
        .ci-dot.green  { background: #22c55e; box-shadow: 0 0 4px #22c55e66; }
        .ci-dot.yellow { background: #f59e0b; box-shadow: 0 0 4px #f59e0b66; }

        /* LAYOUT */
        .main-content { padding: 14px 16px; }
        .page-row { display: flex; gap: 12px; margin-bottom: 12px; }
        .panel { flex: 1; border: 0.5px solid #e2e8f0; padding: 14px; background: #ffffff; border-radius: 8px; min-width: 0; }
        .panel-title { font-size: 9px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.13em; color: #94a3b8; margin-bottom: 12px; display: block; }

        /* FORMS */
        input[type="number"], input[type="text"], select { margin: 3px; padding: 5px 8px; border-radius: 5px; border: 0.5px solid #cbd5e1; background: #f8fafc; color: #0f172a; font-size: 0.875em; }
        input:focus, select:focus { outline: none; border-color: #bfdbfe; }

        /* BUTTONS */
        button { margin: 3px; padding: 7px 12px; cursor: pointer; border-radius: 5px; border: 0.5px solid #e2e8f0; background: #f8fafc; color: #475569; font-size: 0.82em; font-weight: 500; transition: opacity 0.15s; }
        button:hover { opacity: 0.78; }
        .btn-arm    { background: #f0fdf4; color: #16a34a; border-color: #bbf7d0; }
        .btn-disarm { background: #fff1f2; color: #dc2626; border-color: #fecdd3; }
        .btn-action { background: #eff6ff; color: #2563eb; border-color: #bfdbfe; }
        .btn-apply  { background: #eff6ff; color: #2563eb; border-color: #bfdbfe; }
        .btn-clear  { background: #f8fafc; color: #475569; border-color: #e2e8f0; }

        /* CMD GROUPS */
        .cmd-group { font-size: 9px; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; font-weight: 600; margin: 12px 0 5px 3px; }

        /* FENCE PILL TABS */
        .fence-tabs { display: flex; gap: 5px; margin-bottom: 10px; }
        #btnPolygonMode, #btnCircleMode { flex: 1; text-align: center; border-radius: 99px; background: #f8fafc; color: #94a3b8; border-color: #e2e8f0; font-size: 0.8em; padding: 5px 10px; margin: 0; }
        #btnPolygonMode.btn-active, #btnCircleMode.btn-active { background: #eff6ff; color: #2563eb; border-color: #bfdbfe; }

        /* FENCE PANELS */
        .fence-panel { display: none; border: 0.5px dashed #cbd5e1; padding: 10px; border-radius: 6px; margin-top: 8px; background: #ffffff; }
        .fence-panel-title { font-size: 9px; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; font-weight: 600; margin-bottom: 6px; }
        .fence-panel p { font-size: 0.78em; color: #cbd5e1; margin: 4px 0 8px 0; }
        .fence-section { border: 0.5px solid #e2e8f0; border-radius: 5px; padding: 8px 10px; margin-top: 8px; background: #f8fafc; }
        .fence-section-title { font-size: 9px; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; font-weight: 600; margin-bottom: 3px; }
        .fence-section hr { border: none; border-top: 0.5px solid #f1f5f9; margin: 3px 0 6px 0; }
        .fence-status { font-size: 0.82em; color: #475569; min-height: 18px; }

        /* LIVE FEED */
        .live-feed-placeholder { height: 400px; background: #0f172a; border-radius: 5px; color: #475569; display: flex; align-items: center; justify-content: center; text-align: center; padding: 12px; flex-direction: column; gap: 10px; background-image: linear-gradient(#0f1118 1px, transparent 1px), linear-gradient(90deg, #0f1118 1px, transparent 1px); background-size: 28px 28px; }
        .cam-icon { width: 48px; height: 48px; border-radius: 10px; border: 1.5px solid #334155; display: flex; align-items: center; justify-content: center; }
        .cam-icon svg, .cam-icon svg * { stroke: #334155 !important; }
        .cam-disconnect-text { font-size: 0.75em; color: #475569; }
        .live-feed-video { width: 100%; height: 400px; object-fit: cover; border-radius: 5px; display: none; background: #000; }
        .camera-status { font-size: 9px; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 8px; display: block; }

        /* MAP */
        #map { height: 400px; border-radius: 5px; background: #e8f0e9; }
        .map-wrapper { position: relative; }
        .map-wrapper::before {
            content: '';
            position: absolute;
            inset: 0;
            pointer-events: none;
            border-radius: 5px;
            background-image: linear-gradient(#d1dbd2 1px, transparent 1px), linear-gradient(90deg, #d1dbd2 1px, transparent 1px);
            background-size: 28px 28px;
            z-index: 350;
        }
        .map-coords { position: absolute; bottom: 8px; left: 8px; font-size: 10px; font-family: 'Courier New', monospace; color: #64748b; background: rgba(255,255,255,0.88); padding: 2px 7px; border-radius: 3px; border: 0.5px solid #e2e8f0; z-index: 1000; pointer-events: none; }

        /* PULSING DRONE MARKER */
        @keyframes pulse-ring { 0% { transform: scale(0.8); opacity: 0.8; } 70% { transform: scale(2.5); opacity: 0; } 100% { transform: scale(2.5); opacity: 0; } }
        .drone-marker-wrap { position: relative; width: 12px; height: 12px; }
        .drone-dot { position: absolute; top: 0; left: 0; width: 12px; height: 12px; background: #2563eb; border-radius: 50%; border: 2px solid #ffffff; }
        .drone-ring { position: absolute; top: 0; left: 0; width: 12px; height: 12px; border: 2px solid #93c5fd; border-radius: 50%; animation: pulse-ring 1.8s ease-out infinite; }

        /* STATUS ROWS */
        .status-rows { font-size: 0.83em; }
        .status-row { display: flex; justify-content: space-between; align-items: baseline; padding: 5px 0; border-bottom: 0.5px solid #f1f5f9; }
        .status-row:last-child { border-bottom: none; }
        .status-key { color: #94a3b8; }
        .status-val { color: #334155; font-family: 'Courier New', monospace; font-size: 0.95em; }
        .status-val.ok   { color: #16a34a; }
        .status-val.warn { color: #d97706; }
        .status-val.err  { color: #d97706; }

        /* TELEMETRY CARDS */
        .telemetry-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        .telem-card { background: #f8fafc; border: 0.5px solid #e2e8f0; border-radius: 5px; padding: 7px 10px; }
        .telem-label { font-size: 9px; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; margin-bottom: 4px; }
        .telem-val { font-size: 15px; font-family: 'Courier New', monospace; color: #0f172a; }
        .telem-val.cyan   { color: #0284c7; }
        .telem-val.yellow { color: #d97706; }
        .battery-bar-wrap { margin-top: 5px; height: 3px; background: #e2e8f0; border-radius: 2px; overflow: hidden; }
        .battery-bar { height: 100%; background: #f59e0b; border-radius: 2px; transition: width 0.6s; }
        .battery-bar.low      { background: #f59e0b; }
        .battery-bar.critical { background: #d97706; }
        .telem-extra { margin-top: 10px; font-size: 0.8em; }
        .telem-extra-row { display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 0.5px solid #f1f5f9; }
        .telem-extra-row:last-child { border-bottom: none; }
        .telem-extra-key { color: #94a3b8; }
        .telem-extra-val { color: #334155; font-family: 'Courier New', monospace; }
        .telem-extra-val.ok { color: #16a34a; }

        @media (max-width: 700px) { .page-row { flex-direction: column; } }
    </style>
</head>
<body>
    <!-- TOP BAR -->
    <div class="top-bar">
        <div class="logo-dot"></div>
        <span class="app-title">Drone Control</span>
        <span class="badge" id="backendBadge">&#8212;</span>
        <span class="badge" id="gpsBadge">GPS: &#8212;</span>
        <div class="ci-list">
            <div class="ci-item"><div class="ci-dot" id="ciCamera"></div>Camera</div>
            <div class="ci-item"><div class="ci-dot" id="ciTelemetry"></div>Telemetry</div>
            <div class="ci-item"><div class="ci-dot" id="ciLink"></div>Link</div>
            <div class="ci-item"><div class="ci-dot" id="ciArmed"></div>Armed</div>
        </div>
    </div>

    <div class="main-content">

    <!-- Row 1: Live Feed | Map -->
    <div class="page-row">
        <div class="panel">
            <span class="panel-title">Live Feed</span>
            <span class="camera-status" id="cameraConnectionLabel">Camera disconnected</span>
            <div class="live-feed-placeholder" id="liveFeedPlaceholder">
                <div class="cam-icon">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2a2e3a" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
                </div>
                <span class="cam-disconnect-text">Connect laptop to AstaPi Wi-Fi</span>
            </div>
            <video id="liveFeedVideo" class="live-feed-video" muted autoplay playsinline></video>
        </div>
        <div class="panel">
            <span class="panel-title">Map</span>
            <div class="map-wrapper">
                <div id="map"></div>
                <div class="map-coords" id="mapCoords">&mdash;</div>
            </div>
        </div>
    </div>

    <!-- Row 2: Commands | Status + Telemetry -->
    <div class="page-row">
        <div class="panel">
            <span class="panel-title">Commands</span>

            <div class="cmd-group">Flight Control</div>
            <button class="btn-arm"    onclick="arm()">Arm</button>
            <button class="btn-disarm" onclick="disarm()">Disarm</button>

            <div class="cmd-group">Takeoff</div>
            <input type="number" id="takeoffAlt" placeholder="Relative Alt (m)" value="2">
            <button class="btn-action" onclick="takeoff()">Takeoff</button>

            <div class="cmd-group">Hold &amp; Land</div>
            <button onclick="land()">Land</button>
            <button onclick="hold()">Hold</button>

            <div class="cmd-group">Goto</div>
            <input type="number" id="gotoAlt" placeholder="Rel Alt (m)" value="10">
            <div style="font-size:0.78em;color:#475569;margin:6px 0 3px 3px;">Relative target</div>
            <input type="number" id="gotoDistance" placeholder="Distance (m)" value="5" min="0.1" step="0.1">
            <select id="gotoDirection">
                <option value="N">North</option>
                <option value="S">South</option>
                <option value="E">East</option>
                <option value="W">West</option>
            </select>
            <button class="btn-action" onclick="gotoRelative()">Go Relative</button>

            <div class="cmd-group">Geofence</div>
            <div style="font-size:0.78em;color:#475569;margin:0 0 8px 3px;">Set a keep-in boundary. Drone lands or holds if it exits.</div>
            <div class="fence-tabs">
                <button id="btnPolygonMode" onclick="setFenceMode('polygon')">&#9999; Polygon</button>
                <button id="btnCircleMode"  onclick="setFenceMode('circle')" >&#8857; Circular</button>
            </div>

            <div id="polygonFencePanel" class="fence-panel">
                <div class="fence-panel-title">Polygon Fence</div>
                <p>Click the map to add vertices (min 3). A dashed preview appears as you click.</p>
                <div id="polyPointsList" class="fence-status">No points added yet.</div>
                <div style="margin-top:8px;">
                    <button onclick="undoPolyPoint()">&#8617; Undo</button>
                    <button class="btn-apply" onclick="applyPolygonFence()">&#10004; Apply</button>
                    <button class="btn-clear" onclick="clearPolygonFence()">&#10006; Clear</button>
                </div>
            </div>

            <div id="circleFencePanel" class="fence-panel">
                <div class="fence-panel-title">Circular Fence</div>
                <p>Click the map to place the center, then set the radius and apply.</p>
                <div style="font-size:0.83em;color:#94a3b8;margin-bottom:6px;">Center: <span id="circleCenterDisplay" class="fence-status">Not set &mdash; click the map</span></div>
                <div style="margin-top:6px;font-size:0.83em;color:#94a3b8;">Radius: <input type="number" id="circleRadius" value="100" min="1" style="width:70px;"> m</div>
                <div style="margin-top:8px;">
                    <button class="btn-apply" onclick="applyCircularFence()">&#10004; Apply</button>
                    <button class="btn-clear" onclick="clearCircularFence()">&#10006; Clear</button>
                </div>
            </div>

            <div class="fence-section">
                <div class="fence-section-title">Polygon Fences</div><hr>
                <div id="polygonFenceStatus" class="fence-status">None</div>
            </div>
            <div class="fence-section">
                <div class="fence-section-title">Circular Fences</div><hr>
                <div id="circularFenceStatus" class="fence-status">None</div>
            </div>
        </div>

        <div class="panel">
            <span class="panel-title">Status</span>
            <div id="status" class="status-rows">
                <div class="status-row"><span class="status-key">State</span><span class="status-val">&mdash;</span></div>
            </div>
            <span class="panel-title" style="margin-top:16px;display:block;">Telemetry</span>
            <div id="telemetry">
                <div class="telemetry-grid">
                    <div class="telem-card"><div class="telem-label">Altitude</div><div class="telem-val cyan">&mdash;</div></div>
                    <div class="telem-card"><div class="telem-label">Speed</div><div class="telem-val cyan">&mdash;</div></div>
                    <div class="telem-card"><div class="telem-label">Heading</div><div class="telem-val">&mdash;</div></div>
                    <div class="telem-card">
                        <div class="telem-label">Battery</div>
                        <div class="telem-val yellow">&mdash;</div>
                        <div class="battery-bar-wrap"><div class="battery-bar" style="width:0%"></div></div>
                    </div>
                </div>
                <div class="telem-extra">
                    <div class="telem-extra-row"><span class="telem-extra-key">Lat</span><span class="telem-extra-val">&mdash;</span></div>
                    <div class="telem-extra-row"><span class="telem-extra-key">Lon</span><span class="telem-extra-val">&mdash;</span></div>
                    <div class="telem-extra-row"><span class="telem-extra-key">Mode</span><span class="telem-extra-val">&mdash;</span></div>
                    <div class="telem-extra-row"><span class="telem-extra-key">Armed</span><span class="telem-extra-val">&mdash;</span></div>
                    <div class="telem-extra-row"><span class="telem-extra-key">GPS Fix</span><span class="telem-extra-val">&mdash;</span></div>
                    <div class="telem-extra-row"><span class="telem-extra-key">Satellites</span><span class="telem-extra-val">&mdash;</span></div>
                </div>
            </div>
        </div>
    </div>

    </div><!-- /.main-content -->

    <script>
        const HLS_STREAM_URL = 'http://192.168.4.1:8888/camera/index.m3u8';
        const CAMERA_POLL_MS = 1000;

        let map;
        let droneMarker   = null;
        let polygonLayer  = null;
        let circleLayer   = null;
        let fenceMode     = null;
        let polyDraft     = [];
        let draftPolyLayer    = null;
        let circleCenter      = null;
        let draftCenterMarker = null;

        // -- Camera state
        let cameraConnected = false;
        let cameraPollTimer = null;
        let cameraPollInFlight = false;
        let hlsInstance = null;

        function clamp(value, min, max) {
            return Math.min(max, Math.max(min, value));
        }

        function setCameraUiConnected(isConnected) {
            cameraConnected = isConnected;
            const camLabel = document.getElementById('cameraConnectionLabel');
            const ciCamera = document.getElementById('ciCamera');
            if (isConnected) {
                camLabel.textContent = 'Connected to AstaPi Camera';
                if (ciCamera) ciCamera.className = 'ci-dot green';
            } else {
                camLabel.textContent = 'Camera disconnected. Connect laptop to AstaPi Wi-Fi.';
                if (ciCamera) ciCamera.className = 'ci-dot';
            }
        }

        function destroyLiveFeed() {
            if (hlsInstance) {
                try { hlsInstance.destroy(); } catch (e) { console.warn('[Camera] HLS destroy failed', e); }
                hlsInstance = null;
            }
            const video = document.getElementById('liveFeedVideo');
            const placeholder = document.getElementById('liveFeedPlaceholder');
            video.pause();
            video.removeAttribute('src');
            video.load();
            video.style.display = 'none';
            placeholder.style.display = 'flex';
            placeholder.innerHTML = '<div class="cam-icon"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2a2e3a" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg></div><span class="cam-disconnect-text">Connect laptop to AstaPi Wi-Fi</span>';
        }

        function ensureLiveFeed() {
            const video = document.getElementById('liveFeedVideo');
            const placeholder = document.getElementById('liveFeedPlaceholder');
            if (hlsInstance || video.style.display === 'block') return;
            placeholder.style.display = 'none';
            video.style.display = 'block';
            if (window.Hls && Hls.isSupported()) {
                hlsInstance = new Hls();
                hlsInstance.loadSource(HLS_STREAM_URL);
                hlsInstance.attachMedia(video);
                hlsInstance.on(Hls.Events.ERROR, function(_event, data) {
                    if (data && data.fatal) { console.warn('[Camera] HLS fatal error', data); destroyLiveFeed(); }
                });
            } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
                video.src = HLS_STREAM_URL;
            } else {
                placeholder.style.display = 'flex';
                video.style.display = 'none';
                placeholder.innerHTML = '<div class="cam-icon"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2a2e3a" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg></div><span class="cam-disconnect-text">HLS not supported in this browser</span>';
                return;
            }
            video.play().catch(function(err) { console.warn('[Camera] Autoplay blocked or failed', err); });
        }

        async function pollCameraStream() {
            if (cameraPollInFlight) return;
            cameraPollInFlight = true;
            const abortCtrl = new AbortController();
            const abortTimer = setTimeout(() => abortCtrl.abort(), 3000);
            try {
                await fetch(HLS_STREAM_URL, { method: 'GET', mode: 'no-cors', cache: 'no-store', signal: abortCtrl.signal });
                clearTimeout(abortTimer);
                if (!cameraConnected) setCameraUiConnected(true);
                ensureLiveFeed();
            } catch (e) {
                if (cameraConnected) console.warn('[Camera] stream probe failed, marking disconnected', e);
                setCameraUiConnected(false);
                destroyLiveFeed();
            } finally {
                cameraPollInFlight = false;
            }
        }

        function initMap() {
            map = L.map('map').setView([37.7749, -122.4194], 13);
            L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
                subdomains: 'abcd',
                maxZoom: 19
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
                    draftCenterMarker = L.circleMarker(circleCenter, {radius: 7, color: '#22d3ee', fillColor: '#22d3ee', fillOpacity: 0.5, weight: 2}).addTo(map);
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
                draftPolyLayer = L.polyline(polyDraft, {color: '#22d3ee', dashArray: '6,5'}).addTo(map);
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

        function sRow(key, val, cls) {
            return '<div class="status-row"><span class="status-key">' + key + '</span><span class="status-val' + (cls ? ' ' + cls : '') + '">' + val + '</span></div>';
        }

        async function updateStatus() {
            const status = await apiCall('/api/status', 'GET');
            const polyInfo = status.polygon
                ? 'Active (' + (status.polygon.length - 1) + ' vertices)'
                : 'None';
            const circInfo = status.circle
                ? 'Active (r\u202f=\u202f' + status.circle.radius_m + '\u202fm)'
                : 'None';

            let taskRows = '';
            if (status.background_tasks && Object.keys(status.background_tasks).length > 0) {
                for (const [taskId, taskMsg] of Object.entries(status.background_tasks)) {
                    taskRows += sRow(taskId, taskMsg, 'warn');
                }
            }

            const stateVal = status.state || '\u2014';
            const stateStr = stateVal.toLowerCase();
            const stateCls = stateStr.includes('hover') || stateStr.includes('navigat') ? 'ok'
                : stateStr.includes('error') || stateStr.includes('fail') ? 'err'
                : stateStr.includes('hold') ? 'warn' : '';
            const safetyMsg = status.latest_safety_message || 'None';
            const safetyCls = safetyMsg !== 'None' ? 'warn' : '';

            document.getElementById('status').innerHTML =
                sRow('State',    stateVal, stateCls) +
                sRow('Command',  status.current_command || 'None') +
                sRow('Polygon',  polyInfo, status.polygon ? 'ok' : '') +
                sRow('Circular', circInfo, status.circle  ? 'ok' : '') +
                sRow('Safety',   safetyMsg, safetyCls) +
                sRow('Backend',  status.using_mock_drone ? 'Mock' : 'Real MAVLink') +
                taskRows;

            const telem = status.telemetry;
            const bat = telem.battery_percent != null ? parseFloat(telem.battery_percent) : NaN;
            const batPct = isNaN(bat) ? 0 : Math.min(100, Math.max(0, bat));
            const batBarCls = batPct < 20 ? 'critical' : batPct < 40 ? 'low' : '';
            const batValCls = batPct < 40 ? 'yellow' : 'yellow';

            document.getElementById('telemetry').innerHTML =
                '<div class="telemetry-grid">' +
                  '<div class="telem-card"><div class="telem-label">Altitude</div><div class="telem-val cyan">' + (telem.alt != null ? telem.alt + '\u202fm' : '\u2014') + '</div></div>' +
                  '<div class="telem-card"><div class="telem-label">Speed</div><div class="telem-val cyan">' + (telem.speed != null ? telem.speed + '\u202fm/s' : '\u2014') + '</div></div>' +
                  '<div class="telem-card"><div class="telem-label">Heading</div><div class="telem-val">' + (telem.heading != null ? telem.heading + '\u00b0' : '\u2014') + '</div></div>' +
                  '<div class="telem-card"><div class="telem-label">Battery</div><div class="telem-val ' + batValCls + '">' + (telem.battery_percent != null ? telem.battery_percent + '%' : '\u2014') + '</div><div class="battery-bar-wrap"><div class="battery-bar ' + batBarCls + '" style="width:' + batPct + '%"></div></div></div>' +
                '</div>' +
                '<div class="telem-extra">' +
                  '<div class="telem-extra-row"><span class="telem-extra-key">Lat</span><span class="telem-extra-val">' + (telem.lat != null ? telem.lat : '\u2014') + '</span></div>' +
                  '<div class="telem-extra-row"><span class="telem-extra-key">Lon</span><span class="telem-extra-val">' + (telem.lon != null ? telem.lon : '\u2014') + '</span></div>' +
                  '<div class="telem-extra-row"><span class="telem-extra-key">Mode</span><span class="telem-extra-val">' + (telem.mode || '\u2014') + '</span></div>' +
                  '<div class="telem-extra-row"><span class="telem-extra-key">Armed</span><span class="telem-extra-val' + (telem.armed ? ' ok' : '') + '">' + (telem.armed != null ? String(telem.armed) : '\u2014') + '</span></div>' +
                  '<div class="telem-extra-row"><span class="telem-extra-key">GPS Fix</span><span class="telem-extra-val">' + (telem.gps_fix_type != null ? telem.gps_fix_type : '\u2014') + '</span></div>' +
                  '<div class="telem-extra-row"><span class="telem-extra-key">Satellites</span><span class="telem-extra-val">' + (telem.satellites_visible != null ? telem.satellites_visible : '\u2014') + '</span></div>' +
                '</div>';

            // Update top-bar indicators
            const ciTelem = document.getElementById('ciTelemetry');
            const ciLink  = document.getElementById('ciLink');
            const ciArmed = document.getElementById('ciArmed');
            const backendBadge = document.getElementById('backendBadge');
            if (ciTelem) ciTelem.className = 'ci-dot green';
            if (ciLink)  ciLink.className  = 'ci-dot green';
            if (ciArmed) ciArmed.className  = telem.armed ? 'ci-dot green' : 'ci-dot';
            if (backendBadge) backendBadge.textContent = status.using_mock_drone ? 'Mock' : 'Real MAVLink';

            if (telem.lat && telem.lon) {
                const pos = [telem.lat, telem.lon];
                if (!droneMarker) {
                    const icon = L.divIcon({
                        className: '',
                        html: '<div class="drone-marker-wrap"><div class="drone-ring"></div><div class="drone-dot"></div></div>',
                        iconSize: [12, 12],
                        iconAnchor: [6, 6]
                    });
                    droneMarker = L.marker(pos, {icon}).addTo(map);
                } else {
                    droneMarker.setLatLng(pos);
                }
                map.setView(pos);
                const coordsEl = document.getElementById('mapCoords');
                if (coordsEl) coordsEl.textContent = telem.lat.toFixed(6) + ', ' + telem.lon.toFixed(6);
            }

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
        setCameraUiConnected(false);
        destroyLiveFeed();
        pollCameraStream();
        cameraPollTimer = setInterval(pollCameraStream, CAMERA_POLL_MS);

        updateStatus();
        setInterval(updateStatus, 1000);

        window.addEventListener('beforeunload', function() {
            if (cameraPollTimer) {
                clearInterval(cameraPollTimer);
                cameraPollTimer = null;
            }
            destroyLiveFeed();
        });
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
