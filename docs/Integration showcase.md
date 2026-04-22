# Integration Showcase (ATOS + MAVLink)

This branch keeps both systems intact:
- ATOS platform: frontend + communication software
- MAVLink interface: `mavlink_client` UI/API/backend

## Services

From `communication_software/`:

```bash
docker compose up -d --build backend frontend mavlink_backend redis atos isoObject
```

ATOS UI:
- `http://127.0.0.1:8001`

MAVLink UI:
- `http://127.0.0.1:8010`

## Integration Bridge

ATOS backend endpoint forwards MAVLink commands to the MAVLink API:
- `POST /api/v1/integration/mavlink/goto`
- `POST /api/v1/integration/mavlink/takeoff`
- `POST /api/v1/integration/mavlink/hold`
- `POST /api/v1/integration/mavlink/land`

The bridge target is configured by env var:
- `MAVLINK_API_BASE_URL` (default in compose: `http://mavlink_backend:8010`)

## Demo: ATOS -> MAVLink GoTo

1. Start stack with compose command above.
2. Open ATOS UI (`http://127.0.0.1:8001`).
3. Open MAVLink UI (`http://127.0.0.1:8010`) to observe state/telemetry.
4. Trigger bridge call from host shell:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/integration/mavlink/goto \
  -H "Content-Type: application/json" \
  -d '{"latitude":57.7058,"longitude":11.9381,"relative_altitude_m":10}'
```

5. Verify MAVLink UI/API reflects command execution.

Optional detection-triggered forwarding:
- Set `MAVLINK_AUTOTRIGGER_ON_DETECTION=true` in backend env.
- Send detections to `POST /api/v1/set_detections`.
- First detection GPS point forwards automatically to MAVLink `/api/goto`.
