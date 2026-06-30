#!/usr/bin/env bash
# One-shot scheduled start of the trading agent, fired by launchd
# (com.kaushik.tradingagent.scheduledstart) at a specific wall-clock time.
#
# It relaunches the agent via restart.sh (which writes .agent.pid so the normal
# watchdog resumes guarding it), then DELETES ITSELF so it never fires again.
# restart.sh nohup's the agent, so booting out this launchd job can't kill it.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

echo "$(date '+%Y-%m-%d %H:%M:%S') scheduled_start firing" >> logs/scheduled_start.log
./restart.sh >> logs/scheduled_start.log 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') restart.sh done; self-removing one-shot job" >> logs/scheduled_start.log

# Self-destruct: remove the plist first (so it won't reload at next login),
# then boot out the running job (this may terminate this script — fine, the
# agent is already relaunched and nohup'd).
PLIST="$HOME/Library/LaunchAgents/com.kaushik.tradingagent.scheduledstart.plist"
rm -f "$PLIST"
launchctl bootout "gui/$(id -u)/com.kaushik.tradingagent.scheduledstart" 2>/dev/null || true
