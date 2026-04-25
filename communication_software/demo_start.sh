#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ATOS_UI_URL="${ATOS_UI_URL:-http://localhost:8001}"
MAVLINK_UI_URL="${MAVLINK_UI_URL:-http://localhost:8010}"
BACKEND_HEALTH_URL="${BACKEND_HEALTH_URL:-http://localhost:8000/api/v1/health}"
MAVLINK_HEALTH_URL="${MAVLINK_HEALTH_URL:-http://localhost:8010/api/status}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-120}"
REQUIRE_BACKEND_HEALTH="${REQUIRE_BACKEND_HEALTH:-false}"

COMPOSE_FILES=("-f" "docker-compose.yml")
if [[ -f "docker-compose.demo.yml" ]]; then
  COMPOSE_FILES+=("-f" "docker-compose.demo.yml")
fi

log() {
  echo "[demo_start] $*"
}

compose_cmd() {
  docker compose "${COMPOSE_FILES[@]}" "$@"
}

wait_for_url() {
  local name="$1"
  local url="$2"
  local timeout="$3"
  local start
  start="$(date +%s)"

  while true; do
    # Prefer HEAD checks for readiness because some mounted frontends can
    # intermittently fail response body reads while still serving valid headers.
    local code
    code="$(curl -sS -I -o /dev/null -w "%{http_code}" --max-time 8 "$url" 2>/dev/null || true)"
    if [[ ! "$code" =~ ^[23][0-9][0-9]$ ]]; then
      # Fallback to GET for endpoints that do not support HEAD.
      code="$(curl -sS -o /dev/null -w "%{http_code}" --max-time 8 "$url" 2>/dev/null || true)"
    fi
    if [[ "$code" =~ ^[23][0-9][0-9]$ ]]; then
      log "$name is ready ($url)"
      return 0
    fi

    local now elapsed
    now="$(date +%s)"
    elapsed=$((now - start))
    if (( elapsed >= timeout )); then
      log "Timed out waiting for $name ($url)"
      return 1
    fi

    sleep 2
  done
}

open_url() {
  local url="$1"

  if command -v open >/dev/null 2>&1; then
    open "$url" >/dev/null 2>&1 || true
    return 0
  fi

  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1 || true
    return 0
  fi

  if command -v start >/dev/null 2>&1; then
    start "$url" >/dev/null 2>&1 || true
    return 0
  fi

  log "Could not auto-open browser for $url"
}

if ! command -v docker >/dev/null 2>&1; then
  log "Docker is required but not found in PATH."
  exit 1
fi

log "Starting demo services with Docker Compose..."
compose_cmd up -d redis isoObject atos backend frontend mavlink_backend

log "Waiting for UI and MAVLink service readiness..."
wait_for_url "ATOS UI" "$ATOS_UI_URL" "$WAIT_TIMEOUT_SEC"
wait_for_url "MAVLink service" "$MAVLINK_HEALTH_URL" "$WAIT_TIMEOUT_SEC"

backend_ready=true
if ! wait_for_url "Backend API" "$BACKEND_HEALTH_URL" "$WAIT_TIMEOUT_SEC"; then
  backend_ready=false
  log "Backend API did not become healthy in time."
  log "Recent backend logs:"
  compose_cmd logs --tail=40 backend || true
fi

log "Opening ATOS and MAVLink interfaces..."
open_url "$ATOS_UI_URL"
open_url "$MAVLINK_UI_URL"

log "Demo startup complete."
log "ATOS UI:    $ATOS_UI_URL"
log "MAVLink UI: $MAVLINK_UI_URL"
log "Backend API: ${BACKEND_HEALTH_URL%/api/v1/health}"

if [[ "$backend_ready" != true ]]; then
  log "WARNING: Backend API is not healthy, so full ATOS -> MAVLink integration may be degraded."
  if [[ "$REQUIRE_BACKEND_HEALTH" == "true" ]]; then
    exit 1
  fi
fi