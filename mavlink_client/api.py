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
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Drone Control Interface</title>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        *, *::before, *::after { box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            background: #eef2f7;
            color: #1a2a4a;
            min-height: 100vh;
        }

        /* ── Top bar ── */
        .top-bar {
            background: #0d1f3c;
            color: #fff;
            padding: 10px 22px;
            display: flex;
            align-items: center;
            gap: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        }
        .top-bar-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 36px;
            height: 36px;
            background: rgba(255,255,255,0.08);
            border-radius: 6px;
        }
        .top-bar h1 {
            margin: 0;
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: 0.5px;
        }
        .top-bar .subtitle {
            font-size: 0.75rem;
            color: #7aaddb;
            margin-left: auto;
            font-weight: 400;
        }

        /* ── Dashboard wrapper ── */
        .dashboard {
            padding: 16px 18px;
            max-width: 1640px;
            margin: 0 auto;
        }

        /* ── Card ── */
        .card {
            border: 1px solid #cdd8ea;
            border-radius: 8px;
            background: #fff;
            box-shadow: 0 2px 6px rgba(0,0,0,0.07);
            overflow: hidden;
            margin-bottom: 14px;
        }
        .card-header {
            background: #1a6abf;
            color: #fff;
            padding: 7px 14px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.8px;
            text-transform: uppercase;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .card-header .dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: rgba(255,255,255,0.45);
            flex-shrink: 0;
        }
        .card-body {
            padding: 14px;
        }

        /* ── Top row: Live Feed + Map side by side ── */
        .top-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
            margin-bottom: 14px;
        }
        @media (max-width: 900px) {
            .top-row { grid-template-columns: 1fr; }
        }

        /* ── Live feed placeholder ── */
        .live-feed-placeholder {
            background: #111827;
            height: 360px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #4b5563;
            border-radius: 4px;
        }
        .live-feed-placeholder svg { margin-bottom: 10px; }
        .live-feed-placeholder span { font-size: 0.85rem; }

        /* ── Bottom 3-column grid ── */
        .main-row {
            display: grid;
            grid-template-columns: 260px 260px 1fr;
            gap: 14px;
            align-items: start;
        }
        @media (max-width: 1200px) {
            .main-row { grid-template-columns: 1fr 1fr; }
            .right-col { grid-column: 1 / -1; }
        }
        @media (max-width: 700px) {
            .main-row { grid-template-columns: 1fr; }
        }

        /* ── Buttons ── */
        .btn {
            display: inline-block;
            padding: 6px 13px;
            border-radius: 5px;
            border: 1px solid transparent;
            cursor: pointer;
            font-size: 0.82rem;
            font-weight: 600;
            transition: filter 0.12s;
            line-height: 1.4;
        }
        .btn:hover  { filter: brightness(1.1); }
        .btn:active { filter: brightness(0.9); }
        .btn-success { background: #198754; color: #fff; border-color: #146c43; }
        .btn-danger  { background: #dc3545; color: #fff; border-color: #b02a37; }
        .btn-neutral { background: #6c757d; color: #fff; border-color: #565e64; }
        .btn-primary { background: #1a6abf; color: #fff; border-color: #155799; }
        .btn-apply   { background: #198754; color: #fff; border-color: #146c43; }
        .btn-clear   { background: #dc3545; color: #fff; border-color: #b02a37; }
        .btn-active  { background: #1a6abf; color: #fff; border-color: #155799; }

        .btn-row {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 10px;
        }

        /* ── Form controls ── */
        .form-group {
            margin-bottom: 9px;
        }
        .form-group label {
            display: block;
            font-size: 0.73rem;
            color: #6b7280;
            margin-bottom: 3px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        .form-group input,
        .form-group select {
            width: 100%;
            padding: 5px 8px;
            border: 1px solid #cdd8ea;
            border-radius: 4px;
            font-size: 0.83rem;
            color: #1a2a4a;
            background: #fff;
        }
        .form-group input:focus,
        .form-group select:focus {
            outline: none;
            border-color: #1a6abf;
            box-shadow: 0 0 0 2px rgba(26,106,191,0.15);
        }
        .inline-form {
            display: flex;
            gap: 8px;
            align-items: flex-end;
            flex-wrap: wrap;
        }
        .inline-form .form-group { margin-bottom: 0; flex: 1; min-width: 70px; }

        /* ── Section divider ── */
        .section-title {
            font-size: 0.72rem;
            font-weight: 700;
            color: #1a6abf;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin: 10px 0 7px 0;
            padding-bottom: 4px;
            border-bottom: 1px solid #dce8f5;
        }
        .section-title:first-child { margin-top: 0; }

        /* ── Status & Telemetry rows ── */
        .status-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0 12px;
        }
        .status-row {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            padding: 4px 0;
            border-bottom: 1px solid #f0f4f9;
            font-size: 0.81rem;
        }
        .status-row:last-child { border-bottom: none; }
        .status-row .lbl { color: #9ca3af; font-weight: 500; }
        .status-row .val { font-weight: 700; color: #0d1f3c; text-align: right; }

        /* ── Fence panels ── */
        .fence-section {
            border: 1px solid #cdd8ea;
            border-radius: 6px;
            padding: 9px 10px;
            margin-top: 10px;
            background: #f7fafd;
        }
        .fence-section h4 {
            margin: 0 0 4px 0;
            font-size: 0.75rem;
            color: #1a6abf;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        .fence-section hr {
            border: none;
            border-top: 1px solid #dce8f5;
            margin: 4px 0 7px 0;
        }
        .fence-panel {
            display: none;
            border: 1px solid #cdd8ea;
            padding: 10px;
            border-radius: 6px;
            margin-top: 8px;
            background: #fff;
        }
        .fence-panel b  { font-size: 0.83rem; color: #0d1f3c; }
        .fence-panel p  { font-size: 0.78rem; color: #6b7280; margin: 5px 0 9px 0; }
        .fence-status   { font-size: 0.78rem; color: #6b7280; min-height: 18px; }

        /* ── Map ── */
        #map { height: 360px; border-radius: 0 0 6px 6px; }
    </style>
</head>
<body>

<!-- ═══════════════════════ TOP BAR ═══════════════════════ -->
<div class="top-bar">
    <div class="top-bar-icon">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="2.5" fill="#7aaddb"/>
            <path d="M5 5L9.5 9.5M19 5L14.5 9.5M5 19L9.5 14.5M19 19L14.5 14.5" stroke="#7aaddb" stroke-width="1.8" stroke-linecap="round"/>
            <circle cx="3.5" cy="3.5" r="2" fill="white" fill-opacity="0.85"/>
            <circle cx="20.5" cy="3.5" r="2" fill="white" fill-opacity="0.85"/>
            <circle cx="3.5" cy="20.5" r="2" fill="white" fill-opacity="0.85"/>
            <circle cx="20.5" cy="20.5" r="2" fill="white" fill-opacity="0.85"/>
        </svg>
    </div>
    <h1>ATOS Drone Control Interface</h1>
    <span class="subtitle">Single Drone &mdash; Mission Control</span>
</div>

<div class="dashboard">

    <!-- ═══ TOP ROW: LIVE FEED + MAP SIDE BY SIDE ═══ -->
    <div class="top-row">
        <!-- Live Feed -->
        <div class="card" style="margin-bottom:0;">
            <div class="card-header"><span class="dot"></span>Live Feed</div>
            <div class="card-body" style="padding:10px;">
                <div class="live-feed-placeholder">
                    <svg width="44" height="44" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <rect x="2" y="5" width="20" height="14" rx="2" stroke="#374151" stroke-width="1.4"/>
                        <circle cx="12" cy="12" r="3.2" stroke="#374151" stroke-width="1.4"/>
                        <circle cx="17.2" cy="7.8" r="0.9" fill="#374151"/>
                    </svg>
                    <span>No camera connected &mdash; drone video stream will appear here</span>
                </div>
            </div>
        </div>
        <!-- Map -->
        <div class="card" style="margin-bottom:0;">
            <div class="card-header"><span class="dot"></span>Map</div>
            <div style="padding:0;">
                <div id="map"></div>
            </div>
        </div>
    </div>

    <!-- ═══ BOTTOM 3-COLUMN LAYOUT ═══ -->
    <div class="main-row">

        <!-- ── LEFT: Commands + Goto ── -->
        <div>
            <!-- Commands -->
            <div class="card">
                <div class="card-header"><span class="dot"></span>Commands</div>
                <div class="card-body">
                    <div class="section-title">Flight Control</div>
                    <div class="btn-row">
                        <button class="btn btn-success" onclick="arm()">&#9654; Arm</button>
                        <button class="btn btn-danger"  onclick="disarm()">&#9632; Disarm</button>
                        <button class="btn btn-neutral" onclick="hold()">&#9646;&#9646; Hold</button>
                        <button class="btn btn-danger"  onclick="land()">&#9660; Land</button>
                    </div>
                    <div class="section-title">Takeoff</div>
                    <div class="inline-form">
                        <div class="form-group">
                            <label>Relative Altitude (m)</label>
                            <input type="number" id="takeoffAlt" value="2" placeholder="Alt (m)">
                        </div>
                        <button class="btn btn-primary" style="margin-bottom:1px;" onclick="takeoff()">&#8593; Takeoff</button>
                    </div>
                </div>
            </div>

            <!-- Goto -->
            <div class="card">
                <div class="card-header"><span class="dot"></span>Goto</div>
                <div class="card-body">
                    <p style="font-size:0.77rem;color:#9ca3af;margin:0 0 10px 0;">Move relative to current position.</p>
                    <div class="inline-form">
                        <div class="form-group">
                            <label>Altitude (m)</label>
                            <input type="number" id="gotoAlt" value="10" placeholder="Alt (m)">
                        </div>
                        <div class="form-group">
                            <label>Distance (m)</label>
                            <input type="number" id="gotoDistance" value="5" min="0.1" step="0.1">
                        </div>
                    </div>
                    <div class="form-group" style="margin-top:8px;">
                        <label>Direction</label>
                        <select id="gotoDirection">
                            <option value="N">North</option>
                            <option value="S">South</option>
                            <option value="E">East</option>
                            <option value="W">West</option>
                        </select>
                    </div>
                    <div class="btn-row" style="margin-top:8px;">
                        <button class="btn btn-primary" onclick="gotoRelative()">&#8594; Go Relative</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- ── MIDDLE: GeoFence ── -->
        <div>
            <div class="card">
                <div class="card-header"><span class="dot"></span>GeoFence</div>
                <div class="card-body">
                    <p style="font-size:0.77rem;color:#9ca3af;margin:0 0 10px 0;">
                        Set a keep-in boundary. The drone will land or hold if it exits.
                    </p>
                    <div class="btn-row">
                        <button id="btnPolygonMode" class="btn btn-neutral" onclick="setFenceMode('polygon')">&#9999; Polygon</button>
                        <button id="btnCircleMode"  class="btn btn-neutral" onclick="setFenceMode('circle')">&#8857; Circular</button>
                    </div>

                    <div id="polygonFencePanel" class="fence-panel">
                        <b>Polygon Fence</b>
                        <p>Click the map to add vertices (min 3). A dashed preview appears as you click.</p>
                        <div id="polyPointsList" class="fence-status">No points added yet.</div>
                        <div class="btn-row" style="margin-top:9px;">
                            <button class="btn btn-neutral" onclick="undoPolyPoint()">&#8617; Undo</button>
                            <button class="btn btn-apply"   onclick="applyPolygonFence()">&#10004; Apply</button>
                            <button class="btn btn-clear"   onclick="clearPolygonFence()">&#10006; Clear</button>
                        </div>
                    </div>

                    <div id="circleFencePanel" class="fence-panel">
                        <b>Circular Fence</b>
                        <p>Click the map to place the center, then set the radius and apply.</p>
                        <div style="font-size:0.8rem;color:#6b7280;margin-bottom:7px;">
                            Center: <span id="circleCenterDisplay" class="fence-status">Not set &mdash; click the map</span>
                        </div>
                        <div class="form-group">
                            <label>Radius (m)</label>
                            <input type="number" id="circleRadius" value="100" min="1" style="width:110px;">
                        </div>
                        <div class="btn-row" style="margin-top:9px;">
                            <button class="btn btn-apply" onclick="applyCircularFence()">&#10004; Apply</button>
                            <button class="btn btn-clear" onclick="clearCircularFence()">&#10006; Clear</button>
                        </div>
                    </div>

                    <div class="fence-section" style="margin-top:12px;">
                        <h4>Polygon Fences</h4><hr>
                        <div id="polygonFenceStatus" class="fence-status">None</div>
                    </div>
                    <div class="fence-section">
                        <h4>Circular Fences</h4><hr>
                        <div id="circularFenceStatus" class="fence-status">None</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ── RIGHT: Status + Telemetry ── -->
        <div class="right-col">
            <!-- Status -->
            <div class="card">
                <div class="card-header"><span class="dot"></span>Status</div>
                <div class="card-body">
                    <div id="status" class="status-grid">
                        <div class="status-row"><span class="lbl">State</span><span class="val">--</span></div>
                    </div>
                </div>
            </div>

            <!-- Telemetry -->
            <div class="card">
                <div class="card-header"><span class="dot"></span>Telemetry</div>
                <div class="card-body">
                    <div id="telemetry" class="status-grid">
                        <div class="status-row"><span class="lbl">Lat</span><span class="val">--</span></div>
                    </div>
                </div>
            </div>
        </div>
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
            document.getElementById('btnPolygonMode').className = mode === 'polygon' ? 'btn btn-active' : 'btn btn-neutral';
            document.getElementById('btnCircleMode').className  = mode === 'circle'  ? 'btn btn-active' : 'btn btn-neutral';
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
                taskStatusStr = '<div class="status-row" style="grid-column:1/-1"><span class="lbl" style="color:#e07b00;">Background Tasks</span><span class="val" style="font-size:0.76rem;">';
                for (const [taskId, taskMsg] of Object.entries(status.background_tasks)) {
                    taskStatusStr += taskId + ': ' + taskMsg + ' ';
                }
                taskStatusStr += '</span></div>';
            }

            document.getElementById('status').innerHTML =
                '<div class="status-row"><span class="lbl">State</span><span class="val">'         + status.state                                    + '</span></div>' +
                '<div class="status-row"><span class="lbl">Command</span><span class="val">'       + (status.current_command || 'None')               + '</span></div>' +
                '<div class="status-row"><span class="lbl">Polygon Fence</span><span class="val">' + polyInfo                                         + '</span></div>' +
                '<div class="status-row"><span class="lbl">Circular Fence</span><span class="val">'+ circInfo                                         + '</span></div>' +
                '<div class="status-row"><span class="lbl">Safety</span><span class="val">'        + (status.latest_safety_message || 'None')         + '</span></div>' +
                '<div class="status-row"><span class="lbl">Backend</span><span class="val">'       + (status.using_mock_drone ? 'Mock' : 'Real MAVLink') + '</span></div>' +
                taskStatusStr;

            const telem = status.telemetry;
            document.getElementById('telemetry').innerHTML =
                '<div class="status-row"><span class="lbl">Lat</span><span class="val">'         + telem.lat               + '</span></div>' +
                '<div class="status-row"><span class="lbl">Lon</span><span class="val">'         + telem.lon               + '</span></div>' +
                '<div class="status-row"><span class="lbl">Alt</span><span class="val">'         + telem.alt               + ' m</span></div>' +
                '<div class="status-row"><span class="lbl">Heading</span><span class="val">'     + telem.heading           + '\u00b0</span></div>' +
                '<div class="status-row"><span class="lbl">Speed</span><span class="val">'       + telem.speed             + ' m/s</span></div>' +
                '<div class="status-row"><span class="lbl">Battery</span><span class="val">'     + telem.battery_percent   + '%</span></div>' +
                '<div class="status-row"><span class="lbl">Mode</span><span class="val">'        + telem.mode              + '</span></div>' +
                '<div class="status-row"><span class="lbl">Armed</span><span class="val">'       + telem.armed             + '</span></div>' +
                '<div class="status-row"><span class="lbl">GPS Fix</span><span class="val">'     + telem.gps_fix_type      + '</span></div>' +
                '<div class="status-row"><span class="lbl">Satellites</span><span class="val">'  + telem.satellites_visible+ '</span></div>';

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
