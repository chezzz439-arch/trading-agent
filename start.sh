#!/usr/bin/env bash
# Launch the trading agent + monitoring stack with clear per-step status.
# Usage: ./start.sh   (run from the project root)

set -uo pipefail
cd "$(dirname "$0")"
GREEN=$'\033[32m'; RED=$'\033[31m'; YEL=$'\033[33m'; BOLD=$'\033[1m'; NC=$'\033[0m'
ok() { echo "${GREEN}✓${NC} $1"; }
fail() { echo "${RED}✗ $1${NC}"; exit 1; }
step() { echo "${BOLD}[$1] $2${NC}"; }

DATE=$(date -u +%Y%m%d)
LOG="logs/agent_${DATE}.log"
mkdir -p logs

# 1) venv ----------------------------------------------------------------------
step "1/7" "Activating virtualenv"
[ -f venv/bin/activate ] || fail "venv not found — create it and pip install -r requirements.txt"
# shellcheck disable=SC1091
source venv/bin/activate && ok "venv active ($(python --version 2>&1))"

# 2) .env keys -----------------------------------------------------------------
step "2/7" "Validating .env"
[ -f .env ] || fail ".env missing"
for key in ALPACA_API_KEY ALPACA_SECRET_KEY ALPACA_BASE_URL; do
  grep -q "^${key}=.\+" .env || fail "$key missing/empty in .env"
done
ok "required Alpaca keys present"
grep -q "^TELEGRAM_BOT_TOKEN=.\+" .env && ok "Telegram configured" || echo "${YEL}• Telegram not set (alerts disabled)${NC}"

# 3) Alpaca connection ---------------------------------------------------------
step "3/7" "Testing Alpaca connection"
python - <<'PY' || fail "Alpaca connection failed"
from dotenv import load_dotenv; import os
load_dotenv(os.path.join(os.getcwd(), ".env"))
from config import settings
from src.execution.broker import Broker
a = Broker(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY, paper=settings.PAPER).get_account()
assert str(a.status).endswith("ACTIVE"), f"account {a.status}"
print(f"  account {a.status} | equity ${float(a.equity):,.2f} | buying power ${float(a.buying_power):,.2f}")
PY
ok "Alpaca account active"

# 4) Telegram ------------------------------------------------------------------
step "4/7" "Testing Telegram"
python - <<'PY'
from dotenv import load_dotenv; import os
load_dotenv(os.path.join(os.getcwd(), ".env"))
from src.monitoring.telegram_bot import TelegramNotifier
n = TelegramNotifier()
print("  " + ("sent 'Agent starting'" if n.enabled and n.send("*Agent starting* 🚀") else "skipped (not configured)"))
PY
ok "Telegram step done"

# 5) start agent ---------------------------------------------------------------
step "5/7" "Starting agent (background)"
if pgrep -f "python main.py" >/dev/null; then fail "agent already running (use ./stop.sh first)"; fi
# Redirect to a separate console log; main.py's FileHandler already writes $LOG
# (redirecting here too would duplicate every line).
nohup python main.py >> "logs/agent_console_${DATE}.log" 2>&1 &
echo $! > .agent.pid
sleep 2
kill -0 "$(cat .agent.pid)" 2>/dev/null && ok "agent PID $(cat .agent.pid) → $LOG" || fail "agent died on startup (see $LOG)"

# 6) browser -------------------------------------------------------------------
step "6/7" "Opening dashboard"
sleep 4  # give the custom dashboard server a moment to boot
( command -v open >/dev/null && open http://localhost:8765 ) || \
  ( command -v xdg-open >/dev/null && xdg-open http://localhost:8765 ) || \
  echo "${YEL}• open http://localhost:8765 manually${NC}"
ok "dashboard at http://localhost:8765"

# 7) terminal dashboard (foreground) ------------------------------------------
step "7/7" "Live terminal dashboard (Ctrl+C to exit — agent keeps running)"
echo "${BOLD}Agent is live. ./stop.sh to shut everything down.${NC}"
python -m src.monitoring.terminal_dash
