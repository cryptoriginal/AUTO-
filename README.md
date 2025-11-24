# AUTO+ Telegram AI Trading Bot

## Files

- `app.py` – main bot (Telegram + Gemini + Bybit + optional BingX autotrade)
- `requirements.txt` – Python dependencies
- `Procfile` – start command for Render

## Env Vars

Set these in Render:

- `TELEGRAM_BOT_TOKEN` – from BotFather
- `GEMINI_API_KEY` – from Google AI Studio
- `OWNER_CHAT_ID` – your Telegram numeric ID (for autoscan alerts)
- `BINGX_API_KEY` (optional) – enable autotrade
- `BINGX_API_SECRET` (optional)
- `GEMINI_MODEL` (optional) – default `gemini-1.5-pro` (recommended)

## Deploy

1. Push this repo to GitHub.
2. On Render: **New → Web Service →** connect repo.
3. Runtime: Python 3.11 (or 3.10+).
4. Start command: Render auto-detects from `Procfile` (`web: python app.py`).
5. Click Deploy.

Bot uses **polling**, so make sure:
- No webhook is set for this bot (you already deleted it).
- You run **only this one web service** for this bot token.
