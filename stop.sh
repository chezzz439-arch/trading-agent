#!/usr/bin/env bash
# Gracefully shut down the trading agent + monitoring stack.
# Usage: ./stop.sh

set -uo pipefail
cd "$(dirname "$0")"
GREEN=$'\033[32m'; YEL=$'\033[33m'; NC=$'\033[0m'
ok() { echo "${GREEN}✓${NC} $1"; }

[ -f venv/bin/activate ] && source venv/bin/activate

# Telegram "stopped" first (while creds/env still loadable).
python - <<'PY' 2>/dev/null || true
from dotenv import load_dotenv; import os
load_dotenv(os.path.join(os.getcwd(), ".env"))
from src.monitoring.telegram_bot import TelegramNotifier
n = TelegramNotifier()
if n.enabled:
    n.send("*Agent stopped* 🛑")
PY
ok "sent 'Agent stopped' (if Telegram configured)"

# Graceful SIGTERM to main.py (it flattens nothing but exits cleanly + writes report).
if [ -f .agent.pid ] && kill -0 "$(cat .agent.pid)" 2>/dev/null; then
  kill -TERM "$(cat .agent.pid)" && ok "sent SIGTERM to agent PID $(cat .agent.pid)"
  rm -f .agent.pid
else
  pkill -TERM -f "python main.py" && ok "sent SIGTERM to main.py" || echo "${YEL}• no running agent found${NC}"
fi

# Stop the Streamlit dashboard the agent spawned.
pkill -f "streamlit run src/monitoring/dashboard_app.py" 2>/dev/null && ok "stopped Streamlit" || true

sleep 1
ok "shutdown complete"
