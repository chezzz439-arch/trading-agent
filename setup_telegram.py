"""Interactive helper to find your Telegram chat ID and verify the bot.

Usage:
    1. Create a bot with @BotFather on Telegram and copy its token.
    2. Put the token in .env as TELEGRAM_BOT_TOKEN=...
    3. Open a chat with your new bot and send it any message (e.g. "hi").
    4. Run:  python setup_telegram.py
       It reads recent updates, prints your chat ID, offers to write it to .env,
       and sends a confirmation message.
"""

from __future__ import annotations

import os
import sys

import requests
from dotenv import load_dotenv

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def main() -> None:
    load_dotenv(dotenv_path=ENV_PATH)
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        sys.exit("No TELEGRAM_BOT_TOKEN in .env. Create a bot via @BotFather, then add it.")

    me = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=15).json()
    if not me.get("ok"):
        sys.exit(f"Token rejected by Telegram: {me.get('description')}")
    print(f"✓ Bot verified: @{me['result'].get('username')}")

    print("\nSend your bot any message now, then press Enter…")
    input()
    updates = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=15).json()
    results = updates.get("result", [])
    if not results:
        sys.exit("No messages found. Message the bot first, then re-run this script.")

    chat = results[-1]["message"]["chat"]
    chat_id = str(chat["id"])
    name = chat.get("username") or chat.get("first_name") or "you"
    print(f"✓ Found chat ID {chat_id} ({name})")

    existing = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if existing == chat_id:
        print("TELEGRAM_CHAT_ID already set correctly.")
    else:
        ans = input(f"Write TELEGRAM_CHAT_ID={chat_id} to .env? [y/N] ").strip().lower()
        if ans == "y":
            _upsert_env("TELEGRAM_CHAT_ID", chat_id)
            print("✓ .env updated.")

    requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                  data={"chat_id": chat_id, "text": "✅ Trading Agent: Telegram setup complete."},
                  timeout=15)
    print("✓ Confirmation message sent. You're all set.")


def _upsert_env(key: str, value: str) -> None:
    lines = []
    found = False
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as f:
            lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            found = True
            break
    if not found:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f"{key}={value}\n")
    with open(ENV_PATH, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
