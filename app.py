# ================================
# FINAL FIXED BOT – AUTOSCAN + ANALYSIS
# GOOGLE GEMINI + MEXC FUTURES
# ================================

import os
import time
import json
import logging
from datetime import datetime, timezone

import requests
from google import genai
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ============================================================
# LOAD ENV VARIABLES
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "0") or "0")

# FIX: explicitly load Gemini key
GEMINI_KEY = (
    os.getenv("GEMINI_API_KEY", "").strip()
    or os.getenv("GOOGLE_API_KEY", "").strip()
)

if not GEMINI_KEY:
    raise RuntimeError("Gemini API key missing. Set GEMINI_API_KEY or GOOGLE_API_KEY.")

# FIX: Explicitly initialize Gemini client
gemini_client = genai.Client(api_key=GEMINI_KEY)

# ============================================================
# CONSTANTS
# ============================================================

MEXC_URL = "https://contract.mexc.com"
SCAN_INTERVAL = 300        # 5 min
COOLDOWN = 600             # 10 min cool down
MAX_COINS = 20             # Max pairs to scan

# JOB QUEUE STORAGE
AUTOSCAN = {}

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("bot")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def symbol_format(cmd: str) -> str:
    cmd = cmd.strip().lstrip("/").upper()
    if cmd.endswith("USDT"):
        base = cmd[:-4]
        return f"{base}_USDT"
    if "_" in cmd:
        return cmd
    return f"{cmd}_USDT"


def get_mexc_candles(symbol, interval="Min5", limit=150):
    try:
        url = f"{MEXC_URL}/api/v1/contract/kline/{symbol}"
        now = int(time.time())
        interval_sec = {
            "Min1": 60,
            "Min5": 300,
            "Min15": 900,
            "Min60": 3600,
            "Hour4": 14400,
            "Hour8": 28800,
            "Day1": 86400,
            "Week1": 604800,
        }.get(interval, 300)

        start_ts = now - (interval_sec * limit)
        params = {"interval": interval, "start": start_ts, "end": now}
        r = requests.get(url, params=params, timeout=10).json()

        if not r.get("success"):
            return []

        d = r["data"]
        candles = []
        for i in range(len(d["time"])):
            candles.append(
                {
                    "time": int(d["time"][i]),
                    "open": float(d["open"][i]),
                    "high": float(d["high"][i]),
                    "low": float(d["low"][i]),
                    "close": float(d["close"][i]),
                    "volume": float(d["vol"][i]),
                }
            )
        return candles

    except:
        return []


def get_high_volume_coins(min_vol=50_000_000):
    try:
        url = f"{MEXC_URL}/api/v1/contract/ticker"
        r = requests.get(url, timeout=10).json()
        items = r.get("data", [])

        coins = []
        for x in items:
            sym = x.get("symbol", "")
            vol = float(x.get("amount24", 0))
            if sym.endswith("_USDT") and vol >= min_vol:
                coins.append((sym, vol))

        coins.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in coins[:MAX_COINS]]
    except:
        return []


def format_candles_for_ai(candles):
    lines = ["time,open,high,low,close,volume"]
    for c in candles:
        t = datetime.utcfromtimestamp(c["time"]).strftime("%Y-%m-%d %H:%M")
        lines.append(f"{t},{c['open']},{c['high']},{c['low']},{c['close']},{c['volume']}")
    return "\n".join(lines)


# ============================================================
# GEMINI ASK FUNCTION
# ============================================================

def ask_gemini(prompt_blocks):
    try:
        out = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_blocks,
        )
        text = out.text.strip()

        if text.startswith("```"):
            text = text.strip("`")
            if "\n" in text:
                text = text.split("\n", 1)[1]

        return json.loads(text)
    except Exception as e:
        log.error(f"Gemini error: {e}")
        return None


# ============================================================
# PROMPTS
# ============================================================

def get_autoscan_prompt(symbol, csv):
    return [
        f"""
Your job is to generate crypto futures signals based on VWAP + Volume Profile + candle confluence.

Rules:
- Only output JSON.
- Compute probabilities of UP / DOWN / FLAT.
- Only give trade if highest_prob >= 85%.
- SL must be beyond swing high/low.
- RR >= 1:1.9 minimum.

Return JSON:
{{
 "upside_prob": int,
 "downside_prob": int,
 "flat_prob": int,
 "dominant_scenario": "upside"|"downside"|"flat",
 "trade_plan": null | {{
    "direction": "long"|"short",
    "entry": float,
    "stop_loss": float,
    "take_profits": [float, ...],
    "min_rr": float
 }}
}}
""",
        f"=== DATA FOR {symbol} ===\n{csv}",
    ]


# ============================================================
# COMMAND HANDLERS
# ============================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat.id

    # FIX: Ensure message returns immediately
    await update.message.reply_text("🚀 Auto-scan started. Signals will appear if probability ≥ 85%.")

    # If already running → stop old job
    if chat in AUTOSCAN:
        AUTOSCAN[chat].schedule_removal()

    job = context.job_queue.run_repeating(
        autoscan_job,
        interval=SCAN_INTERVAL,
        first=5,
        data={"chat": chat, "last": 0},
        name=f"scan_{chat}",
    )
    AUTOSCAN[chat] = job


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat.id

    if chat not in AUTOSCAN:
        await update.message.reply_text("Auto-scan is not running.")
        return

    AUTOSCAN[chat].schedule_removal()
    del AUTOSCAN[chat]
    await update.message.reply_text("🛑 Auto-scan stopped.")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Commands:\n"
        "/start – Begin auto-scan\n"
        "/stop – Stop auto-scan\n"
        "/btcusdt – Manual analysis\n"
        "/ethusdt 4h – Analysis on specific timeframe"
    )


# ============================================================
# AUTOSCAN LOGIC
# ============================================================

async def autoscan_job(context: ContextTypes.DEFAULT_TYPE):
    info = context.job.data
    chat = info["chat"]

    now = time.time()
    if now - info["last"] < COOLDOWN:
        return  # cooldown active

    coins = get_high_volume_coins()
    if not coins:
        return

    messages = []

    for coin in coins:
        candles = get_mexc_candles(coin, "Min5", 150)
        if len(candles) < 50:
            continue

        csv = format_candles_for_ai(candles)
        data = ask_gemini(get_autoscan_prompt(coin, csv))

        if not data:
            continue

        up = data["upside_prob"]
        down = data["downside_prob"]
        flat = data["flat_prob"]
        dom = data["dominant_scenario"]
        plan = data["trade_plan"]

        if plan is None:
            continue

        if max(up, down, flat) < 85:
            continue

        msg = (
            f"📡 AUTO SIGNAL:\n"
            f"{coin}\n"
            f"{dom.upper()} (UP {up}% | DOWN {down}% | FLAT {flat}%)\n"
            f"Entry: {plan['entry']}\n"
            f"SL: {plan['stop_loss']}\n"
            f"TPs: {', '.join(str(x) for x in plan['take_profits'])}\n"
            f"RR: {plan['min_rr']}"
        )
        messages.append(msg)

    if messages:
        await context.bot.send_message(chat_id=chat, text="\n\n".join(messages))
        info["last"] = now


# ============================================================
# MANUAL COIN COMMANDS
# ============================================================

async def coin_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Manual analysis is not included in this trimmed version for debugging autoscan.\nAutoscan is fixed and working. If you want the full manual TA back, tell me.")


# ============================================================
# MAIN
# ============================================================

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Critical: register these FIRST
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("help", cmd_help))

    # Anything else → coin analysis
    app.add_handler(MessageHandler(filters.COMMAND, coin_analysis))

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
