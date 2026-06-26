#!/usr/bin/env bash
# Safe restart: serialized kill-ALL + single relaunch.
#
# Hardening (2026-06-26): the previous version sent a single SIGTERM via pkill
# and relaunched after a fixed sleep. A hung single-threaded loop ignores TERM,
# so the old process survived and we ended up with TWO agents (happened 3x). It
# also raced the launchd watchdog (every 120s), which could relaunch at the same
# moment. This version:
#   * kills EVERY agent (incl. the Homebrew-framework "Python main.py") with a
#     TERM -> wait -> KILL escalation and VERIFIES none survive before launching
#   * holds a shared mkdir-based lock (macOS has no flock) so the watchdog can't
#     relaunch concurrently — scripts/watchdog.sh takes the same lock
#   * asserts exactly ONE process is running at the end (fails loudly otherwise)
set -uo pipefail
cd "$(dirname "$0")"
GREEN=$'\033[32m'; RED=$'\033[31m'; NC=$'\033[0m'
ok()   { echo "${GREEN}✓${NC} $1"; }
fail() { echo "${RED}✗${NC} $1"; exit 1; }

PIDFILE=".agent.pid"
LOCKDIR=".launch.lock"                  # shared mutex with scripts/watchdog.sh
# Matches "Python main.py" (framework), "python3 main.py", "python main.py".
AGENT_PAT="[Pp]ython[0-9.]* main\.py"

agent_pids() { pgrep -if "$AGENT_PAT" 2>/dev/null || true; }
agent_count() { agent_pids | grep -c . || true; }

# --- shared launch lock: mkdir is atomic; break it if a dead holder left it -- #
acquire_lock() {
  local tries=0 age
  while ! mkdir "$LOCKDIR" 2>/dev/null; do
    age=$(( $(date +%s) - $(stat -f %m "$LOCKDIR" 2>/dev/null || echo 0) ))
    if [ "$age" -ge 120 ]; then
      rm -rf "$LOCKDIR" 2>/dev/null || true   # stale holder died — reclaim
      continue
    fi
    tries=$((tries + 1))
    [ "$tries" -ge 30 ] && fail "could not acquire launch lock (held ${age}s)"
    sleep 1
  done
  trap 'rm -rf "$LOCKDIR" 2>/dev/null' EXIT
}

kill_all_agents() {
  local pids; pids="$(agent_pids)"
  if [ -z "$pids" ]; then ok "no running agent to kill"; return; fi
  # shellcheck disable=SC2086
  kill -TERM $pids 2>/dev/null || true             # graceful first
  for _ in 1 2 3 4 5 6 7 8; do
    [ -z "$(agent_pids)" ] && break
    sleep 1
  done
  pids="$(agent_pids)"
  if [ -n "$pids" ]; then                           # hung loop ignored TERM -> force
    # shellcheck disable=SC2086
    kill -KILL $pids 2>/dev/null || true
    sleep 1
  fi
  pids="$(agent_pids)"
  [ -z "$pids" ] && ok "all agent processes killed" || fail "could not kill: $pids"
}

[ -f venv/bin/activate ] || fail "venv not found"

# 1. Stand the watchdog down for the kill+relaunch window.
rm -f "$PIDFILE"
ok "pidfile removed (watchdog standing down)"

# 2. Serialize against the watchdog, then kill EVERY agent reliably.
acquire_lock
ok "launch lock acquired"
kill_all_agents

# 3. Relaunch a single instance (explicit venv python — no reliance on activate).
DATE="$(date -u +%Y%m%d)"
nohup ./venv/bin/python main.py >> "logs/agent_console_${DATE}.log" 2>&1 &
echo $! > "$PIDFILE"
sleep 3

# 4. Verify the launched process is alive AND that it is the only one.
GOOD="$(cat "$PIDFILE" 2>/dev/null || true)"
kill -0 "$GOOD" 2>/dev/null || fail "agent died on startup — check logs/agent_console_${DATE}.log"
N="$(agent_count)"
[ "$N" -eq 1 ] || fail "expected exactly 1 agent, found $N: $(agent_pids | tr '\n' ' ')"
ok "agent PID $GOOD running (1 process) — dashboard at http://localhost:8765"
