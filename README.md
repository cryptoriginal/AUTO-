# Gemini 2.5 AI Futures Scanner Bot

Telegram bot that:
- Gives AI-based analysis for any Binance USDT-M futures pair on demand.
- Auto-scans market every 5 minutes and pushes high-probability setups.

## Commands

- `/start` – Turn **auto scanner ON** and show help.
- `/stop` – Turn **auto scanner OFF** (manual analysis still works).
- `/btcusdt` – Multi-timeframe analysis for BTCUSDT.
- `/btcusdt 4h` – 4H-focused analysis.

## Scanner rules

- Only scans Binance USDT-M futures with **24h volume ≥ 50M USDT**.
- Only sends signals if:
  - Upside or downside probability ≥ **75%**.
  - Risk:Reward ratio **≥ 1:2.1**.
  - SL is placed at key structural / volume-profile invalidation level.
  - TP1 is realistic & consistent with the probability (TP2 more ambitious but logical).

Signals are sent to `OWNER_CHAT_ID` only.

## Setup

1. Create a Telegram bot via BotFather → get the token.
2. Get a Gemini API key from Google AI Studio.
3. Create a new GitHub repo and add:
   - `app.py`
   - `requirements.txt`
   - `Procfile`
4. On Render:
   - Create a **Background Worker** linked to the repo.
   - Build command: `pip install -r requirements.txt`
   - Start command: `python app.py`
   - Env vars:
     - `TELEGRAM_BOT_TOKEN`
     - `GEMINI_API_KEY`
     - `OWNER_CHAT_ID` (your numeric Telegram ID)
     - `PYTHON_VERSION` = `3.11.6`
5. Disable webhook once:
   - Open `https://api.telegram.org/bot<YOUR_TOKEN>/deleteWebhook` in browser.

Then talk to your bot:

```text
/start
/suiusdt
/btcusdt 1h
