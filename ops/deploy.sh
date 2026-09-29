#!/usr/bin/env bash
# Usage: ops/deploy.sh IMAGE [MODEL_VERSION]  (run from the project folder)
# Replaces the running container; rolls back if /health does not report MODEL_VERSION in time.
set -euo pipefail

IMAGE=$1
EXPECT=${2:-}
NAME=${NAME:-digit-app}
PORT=${PORT:-8000}

run() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  mkdir -p collected logs
  docker run -d --name "$NAME" --restart unless-stopped -p "$PORT:8000" --user "$(id -u):$(id -g)" \
    -v "$PWD/collected:/srv/collected" -v "$PWD/logs:/srv/logs" "$1" >/dev/null
}

healthy() {
  for _ in $(seq 1 30); do
    version=$(curl -sf "localhost:$PORT/health" | python3 -c "import json, sys; print(json.load(sys.stdin)['version'])" 2>/dev/null || true)
    if [ -n "$version" ] && { [ -z "$EXPECT" ] || [ "$version" = "$EXPECT" ]; }; then
      return 0
    fi
    sleep 1
  done
  return 1
}

PREV=$(docker inspect -f '{{.Config.Image}}' "$NAME" 2>/dev/null || true)
run "$IMAGE"
if healthy; then
  echo "live: $IMAGE on port $PORT"
  exit 0
fi

echo "unhealthy: $IMAGE"
if [ -n "$PREV" ]; then
  run "$PREV"
  EXPECT=""
  if healthy; then echo "rolled back to $PREV"; else echo "rollback to $PREV is unhealthy too"; fi
fi
exit 1
