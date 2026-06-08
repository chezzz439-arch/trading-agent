# Deploy the Trading Agent to DigitalOcean (24/7, laptop-free)

Runs `main.py` as a **systemd service** so it auto-starts on boot and auto-restarts
on crash — the Linux replacement for the local `caffeinate + nohup` trick. The
Streamlit dashboard runs as a second service, reached over an SSH tunnel (kept off
the public internet because it exposes the kill-switch/HALT control).

Telegram needs no inbound ports — it's outbound HTTP — so alerts work as soon as
`.env` is in place.

---

## 0. On your laptop — push the latest code

The droplet pulls from GitHub, so push first:

```bash
cd ~/er-wait-oracle/trading-agent
git push origin main
```

(`.env` is gitignored and will NOT be pushed — you copy it over manually in step 4.)

---

## 1. Create the Droplet

DigitalOcean → **Create → Droplet**:

| Setting | Choice | Why |
|---|---|---|
| Image | **Ubuntu 24.04 LTS** | ships Python 3.12 (good wheel coverage for xgboost/scipy/statsmodels) |
| Plan | **Basic → Regular → 2 GB / 1 vCPU ($12/mo)** minimum | 1 GB risks OOM-kill during ML retrain on 173 symbols; 4 GB is safer headroom |
| Region | **NYC1/NYC3** | closest to Alpaca's infra → lowest data/order latency |
| Authentication | **SSH key** (not password) | |
| Hostname | `trading-agent` | |

Note the droplet's **public IP** once it boots.

---

## 2. First login + base hardening

```bash
ssh root@YOUR_DROPLET_IP

# create a non-root user to run the bot
adduser --disabled-password --gecos "" trader
usermod -aG sudo trader
rsync --archive --chown=trader:trader ~/.ssh /home/trader   # copy SSH access

# system packages
apt update && apt -y upgrade
apt -y install python3.12 python3.12-venv python3-pip git ufw

# clock to US market time (logs/sessions read in ET)
timedatectl set-timezone America/New_York

# firewall: SSH only (dashboard goes through an SSH tunnel, not a public port)
ufw allow OpenSSH
ufw --force enable
```

Reconnect as the bot user:

```bash
exit
ssh trader@YOUR_DROPLET_IP
```

---

## 3. Clone + install

```bash
cd ~
git clone https://github.com/chezzz439-arch/trading-agent.git
cd trading-agent

python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt          # ta, scikit-learn, xgboost, statsmodels, scipy, alpaca-py, streamlit…
```

> If a wheel fails to build (rare on 3.12), `sudo apt -y install build-essential python3.12-dev` and retry.

---

## 4. Secrets — copy `.env` to the droplet

**From your laptop** (new terminal):

```bash
scp ~/er-wait-oracle/trading-agent/.env trader@YOUR_DROPLET_IP:~/trading-agent/.env
```

Back **on the droplet**, lock it down and confirm the keys load:

```bash
chmod 600 ~/trading-agent/.env
cd ~/trading-agent && source venv/bin/activate
python -c "from config import settings; \
print('PAPER' if settings.PAPER else 'LIVE', '| keys loaded:', bool(settings.ALPACA_API_KEY))"
```

Make the dashboard NOT auto-spawn from main.py (we run it as its own service):

```bash
sed -i 's/^STREAMLIT_AUTOSTART: bool = True/STREAMLIT_AUTOSTART: bool = False/' config/settings.py
```

Smoke-test the Alpaca connection:

```bash
python -c "from config import settings; from src.execution.broker import Broker; \
a=Broker(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY, paper=settings.PAPER).get_account(); \
print('account', a.status, '| equity', a.equity)"
```

---

## 5. systemd service — the bot (24/7, auto-restart)

```bash
sudo tee /etc/systemd/system/trading-agent.service > /dev/null <<'UNIT'
[Unit]
Description=Algorithmic Trading Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=trader
WorkingDirectory=/home/trader/trading-agent
ExecStart=/home/trader/trading-agent/venv/bin/python main.py
Restart=always
RestartSec=15
# memory guard: restart instead of letting the OOM killer nuke the box
MemoryMax=1700M

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now trading-agent
```

(`load_dotenv()` reads `.env` from `WorkingDirectory`, so no EnvironmentFile is
needed. Bump `MemoryMax` if you sized up the droplet.)

---

## 6. systemd service — the dashboard

```bash
sudo tee /etc/systemd/system/trading-dashboard.service > /dev/null <<'UNIT'
[Unit]
Description=Trading Agent Streamlit Dashboard
After=network-online.target

[Service]
Type=simple
User=trader
WorkingDirectory=/home/trader/trading-agent
ExecStart=/home/trader/trading-agent/venv/bin/streamlit run src/monitoring/dashboard_app.py \
  --server.address 127.0.0.1 --server.port 8501 --server.headless true
Restart=always
RestartSec=15

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now trading-dashboard
```

Bound to `127.0.0.1` on purpose — only reachable through the SSH tunnel below.

---

## 7. Verify it's alive

```bash
systemctl status trading-agent --no-pager
journalctl -u trading-agent -n 30 --no-pager        # expect: "Agent started | mode=PAPER | … min_score=70"
journalctl -u trading-agent -f                       # live tail (Ctrl+C to stop watching; bot keeps running)
```

You should also get the Telegram **"Agent starting"** message.

---

## 8. View the dashboard (from your laptop)

```bash
ssh -L 8501:localhost:8501 trader@YOUR_DROPLET_IP
```

Leave that open, then browse **http://localhost:8501**. Closing the SSH session
only closes the viewer — the bot and dashboard keep running on the droplet.

---

## Day-to-day operations

```bash
# deploy new code (on the droplet)
cd ~/trading-agent && git pull && source venv/bin/activate && pip install -r requirements.txt
sudo systemctl restart trading-agent

sudo systemctl restart trading-agent     # safe restart
sudo systemctl stop trading-agent        # halt trading
systemctl status trading-agent           # health
journalctl -u trading-agent -f           # live logs
journalctl -u trading-agent --since "today"
free -h                                   # watch memory headroom
```

A reboot (`sudo reboot`) brings both services back automatically — that's the
point of `enable`.

---

## Notes / gotchas

- **Paper vs live**: this deploys whatever `ALPACA_BASE_URL` is in your `.env`.
  Keep it on `paper-api.alpaca.markets` until you mean to go live.
- **Watchlist**: `load_watchlist()` falls back to the static 13-symbol list if
  `config/watchlist.json` isn't present; the screener regenerates the full
  173-symbol file on its normal cadence.
- **Cost**: ~$12/mo (2 GB droplet) + $0 data (IEX free feed). Snapshots/backups
  are an optional +20%.
- **Memory**: if `journalctl` shows restarts near the ML-retrain cadence, size the
  droplet up to 4 GB and raise `MemoryMax`.
- **Security**: never expose 8501 publicly — it carries the HALT/kill control. The
  SSH tunnel keeps it private. Don't commit `.env`.
