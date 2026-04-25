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
        body { font-family: Arial, sans-serif; margin: 16px; background: #f5f5f5; }
        .page-row { display: flex; gap: 16px; margin-bottom: 16px; }
        .panel { flex: 1; border: 1px solid #ccc; padding: 12px; background: #fff; border-radius: 6px; min-width: 0; }
        .status { background: #f0f0f0; padding: 10px; margin-bottom: 10px; border-radius: 4px; font-size: 0.9em; }
        button { margin: 3px; padding: 8px 12px; cursor: pointer; border-radius: 4px; border: 1px solid #bbb; }
        input, textarea, select { margin: 3px; padding: 5px; border-radius: 4px; border: 1px solid #bbb; }
        #map { height: 400px; border-radius: 4px; }
        .fence-section { border: 1px solid #ccc; border-radius: 4px; padding: 10px; margin-top: 10px; background: #fafafa; }
        .fence-section h4 { margin: 0 0 4px 0; font-size: 0.95em; }
        .fence-section hr { border: none; border-top: 1px solid #ddd; margin: 4px 0 8px 0; }
        .fence-panel { display: none; border: 1px solid #ddd; padding: 8px; border-radius: 4px; margin-top: 6px; background: #fff; }
        .fence-panel p { font-size: 0.82em; color: #555; margin: 4px 0 8px 0; }
        .btn-apply  { background: #3a8; color: #fff; border-color: #2a7; }
        .btn-clear  { background: #c44; color: #fff; border-color: #b33; }
        .btn-active { background: #3a8; color: #fff; border-color: #2a7; }
        .fence-status { font-size: 0.83em; color: #555; min-height: 18px; }
        .live-feed-placeholder {
            height: 400px;
            background: #000;
            border-radius: 4px;
            color: #ddd;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 12px;
            box-sizing: border-box;
        }
        .live-feed-video {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 4px;
            display: none;
            background: #000;
        }
        .camera-status {
            font-size: 0.84em;
            color: #666;
            margin-bottom: 8px;
        }
        /* Gimbal Control */
        .gimbal-status-bar { font-size: 0.85em; color: #555; background: #f0f0f0; border-radius: 4px; padding: 5px 8px; margin-bottom: 10px; }
        .gimbal-inner { background: #1e2330; border-radius: 6px; padding: 12px; color: #ddd; }
        .gimbal-inner.disabled {
            opacity: 0.45;
            filter: grayscale(0.25);
        }
        .gimbal-inner label { display: block; font-size: 0.8em; color: #aaa; margin: 8px 0 3px; text-transform: uppercase; letter-spacing: 0.04em; }
        .gimbal-slider-row { display: flex; align-items: center; gap: 6px; }
        .gimbal-slider-row span { font-size: 0.78em; color: #888; white-space: nowrap; }
        .gimbal-slider-row input[type=range] { flex: 1; accent-color: #4af; }
        .gimbal-val { font-size: 0.85em; color: #4af; min-width: 28px; text-align: right; }
        .dpad { display: grid; grid-template-columns: 1fr 1fr 1fr; grid-template-rows: 1fr 1fr 1fr; gap: 4px; width: 120px; margin: 4px 0; }
        .dpad button { padding: 7px 0; font-size: 1em; background: #2c3347; color: #ddd; border: 1px solid #3a4460; border-radius: 4px; cursor: pointer; margin: 0; }
        .dpad button:hover { background: #3a4f7a; }
        .dpad .center { background: #2c3347; }
        .zoom-ticks { display: flex; justify-content: space-between; font-size: 0.72em; color: #888; margin-top: 2px; }
        .gimbal-angle-row { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
        .gimbal-angle-row input { width: 60px; background: #2c3347; color: #ddd; border: 1px solid #3a4460; border-radius: 4px; padding: 4px 6px; }
        .gimbal-angle-row input::placeholder { color: #666; }
        .gimbal-btn { background: #2c3347; color: #ddd; border: 1px solid #3a4460; border-radius: 4px; padding: 6px 12px; cursor: pointer; margin: 2px; }
        .gimbal-btn:hover { background: #3a4f7a; }
        .gimbal-btn.active { background: #c44; border-color: #b33; }
        .gimbal-section-label { font-size: 0.8em; color: #aaa; text-transform: uppercase; letter-spacing: 0.04em; margin: 10px 0 4px; display: block; }
    </style>
</head>
<body>
    <h1>Drone Control Interface</h1>

    <div class="page-row">
        <div class="panel">
            <h2>Live Feed</h2>
            <div class="camera-status" id="cameraConnectionLabel">Camera disconnected. Connect laptop to AstaPi Wi-Fi.</div>
            <div class="live-feed-placeholder" id="liveFeedPlaceholder">Camera disconnected</div>
            <video id="liveFeedVideo" class="live-feed-video" muted autoplay playsinline></video>
        </div>
        <div class="panel">
            <h2>Map</h2>
            <div id="map"></div>
        </div>
    </div>

    <!-- Row 2: Gimbal Control | Commands/GeoFence -->
    <div class="page-row">
        <!-- Gimbal Control panel -->
        <div class="panel">
            <h2>Gimbal Control</h2>
            <div class="camera-status" id="gimbalConnectionLabel">Camera disconnected. Connect laptop to AstaPi Wi-Fi.</div>
            <div class="gimbal-status-bar" id="gimbalStatusBar">Yaw: 0 &nbsp;|&nbsp; Pitch: 0 &nbsp;|&nbsp; Roll: 0</div>
            <div class="gimbal-inner disabled" id="gimbalControlInner">

                <label>Speed</label>
                <div class="gimbal-slider-row">
                    <span>Slow</span>
                    <input type="range" id="gimbalSpeed" min="1" max="100" value="50"
                           oninput="document.getElementById('gimbalSpeedVal').textContent=this.value; gimbalLog('speed',this.value)">
                    <span>Fast</span>
                    <span class="gimbal-val" id="gimbalSpeedVal">50</span>
                </div>

                <label>Pan / Tilt</label>
                <div class="dpad">
                    <div></div>
                    <button onclick="gimbalNudge(0, 2)">&#9650;</button>
                    <div></div>
                    <button onclick="gimbalNudge(-2, 0)">&#9664;</button>
                    <button class="center" onclick="gimbalCenter()">&#9678;</button>
                    <button onclick="gimbalNudge(2, 0)">&#9654;</button>
                    <div></div>
                    <button onclick="gimbalNudge(0, -2)">&#9660;</button>
                    <div></div>
                </div>

                <label>Zoom</label>
                <div class="gimbal-slider-row">
                    <input type="range" id="gimbalZoom" min="1" max="6" step="1" value="1"
                           oninput="onZoomInput(this.value)"
                           onchange="gimbalSetZoom(this.value)">
                </div>
                <div class="zoom-ticks"><span>1x</span><span>2x</span><span>3x</span><span>4x</span><span>5x</span><span>6x</span></div>
                <div style="font-size:0.82em;color:#4af;margin-top:2px;">Zoom: <span id="gimbalZoomVal">1.0x</span></div>

                <label>Absolute Angle</label>
                <div class="gimbal-angle-row">
                    <input type="number" id="gimbalYawInput" placeholder="Yaw" style="">
                    <input type="number" id="gimbalPitchInput" placeholder="Pitch" style="">
                    <button class="gimbal-btn" onclick="gimbalGoAbsolute()">Go</button>
                </div>

                <label>Camera</label>
                <div>
                    <button class="gimbal-btn" onclick="gimbalTakePhoto()">&#128247; Photo</button>
                    <button class="gimbal-btn" id="gimbalRecordBtn" onclick="gimbalToggleRecord()">&#9210; Record</button>
                </div>

                <label>Scan Patterns</label>
                <div>
                    <button class="gimbal-btn" onclick="gimbalRunScan('patrol')">&#8635; Patrol Scan</button>
                    <button class="gimbal-btn" onclick="gimbalRunScan('horizon')">&#8646; Horizon Sweep</button>
                    <button class="gimbal-btn" onclick="gimbalRunScan('nod')">&#8693; Nod Search</button>
                </div>

            </div>
        </div>

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
    </div>

    <!-- Row 3: Status | Telemetry -->
    <div class="page-row">
        <div class="panel">
            <h2>Status</h2>
            <div id="status" class="status">Loading...</div>
        </div>
        <div class="panel">
            <h2>Telemetry</h2>
            <div id="telemetry" class="status">Loading...</div>
        </div>
    </div>

    <script>
        const GIMBAL_BASE_URL = 'http://192.168.4.1:5000';
        const HLS_STREAM_URL = 'http://192.168.4.1:8888/camera/index.m3u8';
        const GIMBAL_POLL_MS = 1000;
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

        // ── Gimbal and Camera integration state ─────────────────────────
        let gimbalRecording = false;
        let cameraConnected = false;
        let gimbalConnected = false;
        let gimbalAngles = { yaw: 0, pitch: 0, roll: 0 };
        let cameraPollTimer = null;
        let gimbalPollTimer = null;
        let gimbalPollInFlight = false;
        let cameraPollInFlight = false;
        let hlsInstance = null;

        function clamp(value, min, max) {
            return Math.min(max, Math.max(min, value));
        }

        function setCameraUiConnected(isConnected) {
            cameraConnected = isConnected;
            const camLabel = document.getElementById('cameraConnectionLabel');

            if (isConnected) {
                camLabel.textContent = 'Connected to AstaPi Camera';
            } else {
                camLabel.textContent = 'Camera disconnected. Connect laptop to AstaPi Wi-Fi.';
            }
        }

        function setGimbalUiConnected(isConnected) {
            gimbalConnected = isConnected;
            const gimbalInner = document.getElementById('gimbalControlInner');
            const gimbalLabel = document.getElementById('gimbalConnectionLabel');
            const controls = gimbalInner.querySelectorAll('button, input, select, textarea');

            controls.forEach((el) => {
                el.disabled = !isConnected;
            });
            gimbalInner.classList.toggle('disabled', !isConnected);

            if (isConnected) {
                gimbalLabel.textContent = 'Connected to AstaPi Camera';
            } else {
                gimbalLabel.textContent = 'Camera disconnected. Connect laptop to AstaPi Wi-Fi.';
            }
        }

        function updateGimbalAnglesDisplay(yaw, pitch, roll) {
            document.getElementById('gimbalStatusBar').textContent =
                'Yaw: ' + Number(yaw).toFixed(1) + ' | Pitch: ' + Number(pitch).toFixed(1) + ' | Roll: ' + Number(roll).toFixed(1);
        }

        function destroyLiveFeed() {
            if (hlsInstance) {
                try {
                    hlsInstance.destroy();
                } catch (e) {
                    console.warn('[Camera] HLS destroy failed', e);
                }
                hlsInstance = null;
            }

            const video = document.getElementById('liveFeedVideo');
            const placeholder = document.getElementById('liveFeedPlaceholder');
            video.pause();
            video.removeAttribute('src');
            video.load();
            video.style.display = 'none';
            placeholder.style.display = 'flex';
            placeholder.textContent = 'Camera disconnected';
        }

        function ensureLiveFeed() {
            const video = document.getElementById('liveFeedVideo');
            const placeholder = document.getElementById('liveFeedPlaceholder');

            if (hlsInstance || video.style.display === 'block') {
                return;
            }

            placeholder.style.display = 'none';
            video.style.display = 'block';

            if (window.Hls && Hls.isSupported()) {
                hlsInstance = new Hls();
                hlsInstance.loadSource(HLS_STREAM_URL);
                hlsInstance.attachMedia(video);
                hlsInstance.on(Hls.Events.ERROR, function(_event, data) {
                    if (data && data.fatal) {
                        console.warn('[Camera] HLS fatal error', data);
                        destroyLiveFeed();
                    }
                });
            } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
                video.src = HLS_STREAM_URL;
            } else {
                placeholder.style.display = 'flex';
                video.style.display = 'none';
                placeholder.textContent = 'HLS not supported in this browser';
                return;
            }

            video.play().catch(function(err) {
                console.warn('[Camera] Autoplay blocked or failed', err);
            });
        }

        async function gimbalApiGet(path, queryParams = {}, allowWhenDisconnected = false) {
            if (!allowWhenDisconnected && !gimbalConnected) {
                return null;
            }

            const url = new URL(GIMBAL_BASE_URL + path);
            Object.entries(queryParams).forEach(([key, value]) => {
                url.searchParams.set(key, String(value));
            });

            const response = await fetch(url.toString(), { method: 'GET' });
            if (!response.ok) {
                throw new Error('HTTP ' + response.status);
            }

            const contentType = response.headers.get('content-type') || '';
            if (contentType.includes('application/json')) {
                return await response.json();
            }
            return null;
        }

        async function pollCameraStream() {
            if (cameraPollInFlight) {
                return;
            }
            cameraPollInFlight = true;

            const abortCtrl = new AbortController();
            const abortTimer = setTimeout(() => abortCtrl.abort(), 3000);

            try {
                await fetch(HLS_STREAM_URL, {
                    method: 'GET',
                    mode: 'no-cors',
                    cache: 'no-store',
                    signal: abortCtrl.signal,
                });
                clearTimeout(abortTimer);

                if (!cameraConnected) {
                    setCameraUiConnected(true);
                }
                ensureLiveFeed();
            } catch (e) {
                if (cameraConnected) {
                    console.warn('[Camera] stream probe failed, marking disconnected', e);
                }
                setCameraUiConnected(false);
                destroyLiveFeed();
            } finally {
                cameraPollInFlight = false;
            }
        }

        async function pollGimbalAngles() {
            if (gimbalPollInFlight) {
                return;
            }
            gimbalPollInFlight = true;

            try {
                const data = await gimbalApiGet('/angles', {}, true);
                if (!data) {
                    throw new Error('No angle payload');
                }

                gimbalAngles.yaw = clamp(Number(data.yaw ?? 0), -45, 45);
                gimbalAngles.pitch = clamp(Number(data.pitch ?? 0), -90, 25);
                gimbalAngles.roll = Number(data.roll ?? 0);
                updateGimbalAnglesDisplay(gimbalAngles.yaw, gimbalAngles.pitch, gimbalAngles.roll);

                if (!gimbalConnected) {
                    setGimbalUiConnected(true);
                }
            } catch (e) {
                if (gimbalConnected) {
                    console.warn('[Gimbal] Poll failed, marking disconnected', e);
                }
                setGimbalUiConnected(false);
            } finally {
                gimbalPollInFlight = false;
            }
        }

        function onZoomInput(value) {
            const zoom = clamp(Number(value), 1, 6);
            document.getElementById('gimbalZoomVal').textContent = zoom.toFixed(1) + 'x';
        }

        function zoomToBackendLevel(zoom) {
            const clampedZoom = clamp(Number(zoom), 1, 6);
            return clamp(Math.round(clampedZoom) * 10, 10, 60);
        }

        function gimbalLog(action, value) {
            console.log('[Gimbal]', action, value);
        }

        async function gimbalSetAngle(yaw, pitch) {
            if (!gimbalConnected) {
                return;
            }

            const targetYaw = clamp(Number(yaw), -45, 45);
            const targetPitch = clamp(Number(pitch), -90, 25);

            try {
                await gimbalApiGet('/set_angle', { yaw: targetYaw, pitch: targetPitch });
                gimbalAngles.yaw = targetYaw;
                gimbalAngles.pitch = targetPitch;
                updateGimbalAnglesDisplay(gimbalAngles.yaw, gimbalAngles.pitch, gimbalAngles.roll);
            } catch (e) {
                console.warn('[Gimbal] set_angle failed', e);
            }
        }

        async function gimbalNudge(deltaYaw, deltaPitch) {
            await gimbalSetAngle(gimbalAngles.yaw + deltaYaw, gimbalAngles.pitch + deltaPitch);
        }

        async function gimbalCenter() {
            if (!gimbalConnected) {
                return;
            }
            try {
                await gimbalApiGet('/center');
            } catch (e) {
                console.warn('[Gimbal] center failed', e);
            }
        }

        function gimbalGoAbsolute() {
            const yaw = Number(document.getElementById('gimbalYawInput').value);
            const pitch = Number(document.getElementById('gimbalPitchInput').value);
            if (!Number.isFinite(yaw) || !Number.isFinite(pitch)) {
                return;
            }
            gimbalSetAngle(yaw, pitch);
        }

        async function gimbalSetZoom(zoomValue) {
            if (!gimbalConnected) {
                return;
            }
            const level = zoomToBackendLevel(zoomValue);
            try {
                await gimbalApiGet('/zoom_abs', { level: level });
            } catch (e) {
                console.warn('[Gimbal] zoom_abs failed', e);
            }
        }

        async function gimbalTakePhoto() {
            if (!gimbalConnected) {
                return;
            }
            try {
                await gimbalApiGet('/photo');
            } catch (e) {
                console.warn('[Gimbal] photo failed', e);
            }
        }

        async function gimbalRunScan(scanType) {
            if (!gimbalConnected) {
                return;
            }

            const validScan = {
                patrol: '/patrol',
                horizon: '/horizon',
                nod: '/nod',
            };

            const endpoint = validScan[scanType];
            if (!endpoint) {
                return;
            }

            try {
                await gimbalApiGet(endpoint);
            } catch (e) {
                console.warn('[Gimbal] scan failed', scanType, e);
            }
        }

        function gimbalToggleRecord() {
            if (!gimbalConnected) {
                return;
            }
            gimbalRecording = !gimbalRecording;
            const btn = document.getElementById('gimbalRecordBtn');

            const endpoint = gimbalRecording ? '/record_start' : '/record_stop';
            gimbalApiGet(endpoint)
                .then(function() {
                    btn.textContent = gimbalRecording ? '\u23F9 Stop' : '\u23FA Record';
                    btn.classList.toggle('active', gimbalRecording);
                })
                .catch(function(e) {
                    gimbalRecording = !gimbalRecording;
                    console.warn('[Gimbal] record toggle failed', e);
                });
        }
        // ────────────────────────────────────────────────────────────────

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
        setCameraUiConnected(false);
        setGimbalUiConnected(false);
        destroyLiveFeed();

        pollCameraStream();
        cameraPollTimer = setInterval(pollCameraStream, CAMERA_POLL_MS);
        pollGimbalAngles();
        gimbalPollTimer = setInterval(pollGimbalAngles, GIMBAL_POLL_MS);

        updateStatus();
        setInterval(updateStatus, 1000);

        window.addEventListener('beforeunload', function() {
            if (cameraPollTimer) {
                clearInterval(cameraPollTimer);
                cameraPollTimer = null;
            }
            if (gimbalPollTimer) {
                clearInterval(gimbalPollTimer);
                gimbalPollTimer = null;
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
