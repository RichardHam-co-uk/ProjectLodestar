#!/bin/bash
# Start LiteLLM router (HTTP proxy on port 4000) and wait for it to be ready.
#
# NOTE: This script is intended to be run on your workstation or the T600 VM
# directly — NOT inside the Cowork / Claude sandbox, which blocks loopback
# TCP connections.  For routing within the Cowork sandbox, use the Python
# library path (litellm.completion + SemanticRouter) instead.

# Resolve repo root relative to this script (works from any CWD)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# Load API keys from ~/.env if it exists
if [ -f "$HOME/.env" ]; then
    set -a
    source "$HOME/.env"
    set +a
    echo "🔑 Loaded API keys from ~/.env"
else
    echo "⚠️  No ~/.env found — cloud provider calls will fail"
fi

echo "🚀 Starting LiteLLM router..."

# Kill any existing router
pkill -f "proxy_server.*4000\|uvicorn.*4000" 2>/dev/null || true
sleep 1

# Create log directory
mkdir -p .lodestar

# Activate venv
source "$REPO_DIR/.venv/bin/activate"

# Launch via uvicorn directly — bypasses the litellm CLI which hangs on startup.
# CONFIG_FILE_PATH is read by litellm's proxy_server startup event.
export CONFIG_FILE_PATH="$REPO_DIR/config/litellm_config.yaml"
export LITELLM_TELEMETRY="False"
export LITELLM_LOG="WARNING"

nohup uvicorn litellm.proxy.proxy_server:app \
              --host 127.0.0.1 \
              --port 4000 \
              --log-level warning \
              > .lodestar/router.log 2>&1 &

ROUTER_PID=$!
echo "$ROUTER_PID" > .lodestar/router.pid

echo "⏳ Waiting for router to be ready..."

# Wait up to 25 seconds for router to respond
for i in {1..25}; do
    if curl -s http://localhost:4000/v1/models > /dev/null 2>&1; then
        echo "✅ Router is ready! (PID: $ROUTER_PID)"
        echo "📋 Log: $REPO_DIR/.lodestar/router.log"
        echo "🛑 Stop with: kill $ROUTER_PID"
        exit 0
    fi
    sleep 1
    echo -n "."
done

echo ""
echo "❌ Router failed to become ready after 25 seconds"
echo "Last 20 lines of log:"
tail -20 .lodestar/router.log
exit 1
