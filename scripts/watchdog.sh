#!/usr/bin/env bash
# Watchdog — restarts the trading agent if it dies or hangs.
#
# Run periodically by launchd (com.kaushik.tradingagent.watchdog, every 120s).
# The agent is single-threaded; an un-timed-out network call can wedge the whole
# loop while the process stays alive (it happened: 53 min frozen, ignored SIGTERM,
# left positions unmonitored). This catches both a dead process and a live-but-
# stuck one, and relaunches it the same way start.sh does.
#
# Contract:
#   .agent.pid PRESENT  -> the agent is supposed to be running; restart if it is
#                          dead or its state file has gone stale (hung).
#   .agent.pid ABSENT   -> stop.sh removed it (deliberate shutdown); do nothing.
# The dashboard HALT button doesn't stop the process — it keeps scanning and
# writing state — so a halted agent looks healthy here and is left alone.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

STATE="logs/agent_state.json"
PIDFILE=".agent.pid"
LOG="logs/watchdog.log"
LOCKDIR=".launch.lock"           # shared mutex with restart.sh — prevents
                                 # concurrent kill+relaunch (duplicate agents)
AGENT_PAT="[Pp]ython[0-9.]* main\.py"
# Stale threshold: 2.5x the 240s scan interval, matching the dashboard's own
# "online" cutoff. A healthy agent rewrites state every scan (~4 min).
STALE_SECS=600

mkdir -p logs
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG"; }

# Shared launch lock (mkdir is atomic; macOS has no flock). Serializes against a
# manual ./restart.sh so the two can't both relaunch and create duplicates.
acquire_lock() {
  local tries=0 age
  while ! mkdir "$LOCKDIR" 2>/dev/null; do
    age=$(( $(date +%s) - $(stat -f %m "$LOCKDIR" 2>/dev/null || echo 0) ))
    if [ "$age" -ge 120 ]; then rm -rf "$LOCKDIR" 2>/dev/null || true; continue; fi
    tries=$((tries + 1))
    [ "$tries" -ge 30 ] && { log "could not acquire launch lock (held ${age}s) — standing down"; exit 0; }
    sleep 1
  done
  trap 'rm -rf "$LOCKDIR" 2>/dev/null' EXIT
}

# Deliberate shutdown (stop.sh removed the pid) -> stand down.
[ -f "$PIDFILE" ] || exit 0
PID="$(cat "$PIDFILE" 2>/dev/null || true)"

alive() { [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; }

# Age (seconds) of the state file's updated_at, or a large number if unreadable.
state_age() {
  [ -f "$STATE" ] || { echo 999999; return; }
  ./venv/bin/python - "$STATE" <<'PY' 2>/dev/null || echo 999999
import json, sys
from datetime import datetime, timezone
try:
    u = json.load(open(sys.argv[1])).get("updated_at")
    print(int((datetime.now(timezone.utc) - datetime.fromisoformat(u)).total_seconds()))
except Exception:
    print(999999)
PY
}

reason=""
if ! alive; then
  reason="process $PID not running"
else
  age="$(state_age)"
  if [ "$age" -gt "$STALE_SECS" ]; then
    reason="hung — state ${age}s stale (>${STALE_SECS}s), pid $PID alive"
  fi
fi

# Healthy -> nothing to do.
[ -n "$reason" ] || exit 0

# Take the shared lock before touching processes so we can't race a manual
# ./restart.sh. If restart.sh is mid-relaunch we'll block here until it's done.
acquire_lock

# Re-check under the lock: restart.sh (or a prior watchdog) may have already
# relaunched a healthy agent while we were waiting — if so, stand down.
PID="$(cat "$PIDFILE" 2>/dev/null || true)"
if alive; then
  age="$(state_age)"
  if [ "$age" -le "$STALE_SECS" ]; then
    log "stood down — agent pid $PID healthy after acquiring lock (state ${age}s)"
    exit 0
  fi
fi

log "RESTART: $reason"

# Telegram heads-up (best effort; never blocks the restart).
./venv/bin/python - "$reason" <<'PY' >/dev/null 2>&1 || true
import os, sys
from dotenv import load_dotenv
load_dotenv(os.path.join(os.getcwd(), ".env"))
from src.monitoring.telegram_bot import TelegramNotifier
n = TelegramNotifier()
if n.enabled:
    n.send(f"\U0001F6A8 *Watchdog* restarting agent\n{sys.argv[1]}")
PY

# Stop the old process if it's still around (hung processes may ignore TERM).
if alive; then
  kill -TERM "$PID" 2>/dev/null
  for _ in 1 2 3 4 5 6 7 8; do alive || break; sleep 1; done
  alive && { kill -KILL "$PID" 2>/dev/null; log "force-killed $PID"; }
fi
# Sweep any stray agents (incl. framework "Python main.py") so we never end up
# with duplicates: TERM, wait, then KILL whatever ignored it.
strays() { pgrep -if "$AGENT_PAT" 2>/dev/null || true; }
if [ -n "$(strays)" ]; then
  # shellcheck disable=SC2046
  kill -TERM $(strays) 2>/dev/null || true
  for _ in 1 2 3 4 5; do [ -z "$(strays)" ] && break; sleep 1; done
  # shellcheck disable=SC2046
  [ -n "$(strays)" ] && { kill -KILL $(strays) 2>/dev/null || true; sleep 1; }
fi

# Relaunch in the background (mirrors start.sh step 5).
DATE="$(date -u +%Y%m%d)"
nohup ./venv/bin/python main.py >> "logs/agent_console_${DATE}.log" 2>&1 &
echo $! > "$PIDFILE"
sleep 3
if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  log "relaunched agent pid $(cat "$PIDFILE")"
else
  log "ERROR: relaunch failed — agent died on startup"
fi
