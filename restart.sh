#!/usr/bin/env bash
# Safe restart: removes .agent.pid so the watchdog stands down during the
# kill+relaunch window, preventing it from spawning a duplicate process.
set -uo pipefail
cd "$(dirname "$0")"
GREEN=$'\033[32m'; BOLD=$'\033[1m'; NC=$'\033[0m'
ok() { echo "${GREEN}✓${NC} $1"; }

[ -f venv/bin/activate ] || { echo "venv not found"; exit 1; }
source venv/bin/activate

# 1. Remove pid file first — watchdog exits immediately when pidfile is absent.
rm -f .agent.pid
ok "pidfile removed (watchdog standing down)"

# 2. Kill any running agent processes.
pkill -if "python main.py" 2>/dev/null && ok "killed existing agent(s)" || true
sleep 2

# 3. Relaunch.
DATE=$(date -u +%Y%m%d)
nohup python main.py >> "logs/agent_console_${DATE}.log" 2>&1 &
echo $! > .agent.pid
sleep 3
if kill -0 "$(cat .agent.pid)" 2>/dev/null; then
    ok "agent PID $(cat .agent.pid) running — dashboard at http://localhost:8765"
else
    echo "ERROR: agent died on startup — check logs/agent_${DATE}.log"
    exit 1
fi
