# FULL APP.PY — FIXED FOR RENDER + PTB3 + ASYNC SCANNER
# ------------------------------------------------------

import os
import json
import time
import asyncio
from datetime import datetime, timezone

import requests
import google.generativeai as genai
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from bingx.api import BingxAPI


# ============================================================
# ENV VARS
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "0"))

BINGX_API_KEY = os.getenv("BINGX_API_KEY")
BINGX_API_SECRET = os.getenv("BINGX_API_SECRET")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")

GEMINI_MODEL = "gemini-1.5-flash"

SCAN_INTERVAL_SECONDS = 300
SIGNAL_COOLDOWN_SECONDS = 600
MIN_PROB_SCAN = 80
MIN_RR = 1.9
MIN_VOLUME = 50_000_000

AUTO_LEVERAGE = 3
AUTO_MAX_POSITIONS = 2

SCAN_ENABLED = True
SUPPORTED_BINGX = set()
auto_open_positions = set()
last_signal_time = {}

# ============================================================
# CLIENTS
# ============================================================

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

bingx = None
if BINGX_API_KEY and BINGX_API_SECRET:
    bingx = BingxAPI(BINGX_API_KEY, BINGX_API_SECRET)


# ============================================================
# BYBIT MARKET DATA
# ============================================================

BYBIT_ENDPOINT = "https://api.bybit.com"

INTERVAL_MAP = {
    "15m": "15",
    "1h": "60",
    "4h": "240",
}


def get_bybit_symbols():
    try:
        r = requests.get(f"{BYBIT_ENDPOINT}/v5/market/tickers",
                         params={"category": "linear"},
                         timeout=10)
        data = r.json()
        lst = data.get("result", {}).get("list", [])
        lst = [x for x in lst if float(x["turnover24h"]) >= MIN_VOLUME]
        lst.sort(key=lambda x: float(x["turnover24h"]), reverse=True)
        return [x["symbol"] for x in lst[:20]]
    except:
        return []


def get_candles(symbol, tf):
    try:
        r = requests.get(f"{BYBIT_ENDPOINT}/v5/market/kline",
                         params={"category": "linear", "symbol": symbol, "interval": INTERVAL_MAP[tf], "limit": 200},
                         timeout=10)
        data = r.json()
        if data["retCode"] != 0:
            return []
        out = []
        for c in data["result"]["list"]:
            out.append({
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            })
        return list(reversed(out))
    except:
        return []


def get_price(symbol):
    try:
        r = requests.get(f"{BYBIT_ENDPOINT}/v5/market/tickers",
                         params={"category": "linear", "symbol": symbol},
                         timeout=10)
        d = r.json()
        x = d.get("result", {}).get("list", [])
        if not x:
            return None
        return float(x[0]["lastPrice"])
    except:
        return None


# ============================================================
# GEMINI
# ============================================================

def run_gemini(prompt):
    try:
        resp = model.generate_content(prompt)
        return resp.text or ""
    except:
        return ""


def extract_json(txt):
    try:
        s = txt.index("{")
        e = txt.rindex("}")
        return json.loads(txt[s:e+1])
    except:
        return None


# ============================================================
# SCAN PROMPT
# ============================================================

def scan_prompt(symbol, candles, price):
    return f"""
You are an expert crypto futures analyst.

Analyze the snapshot:

Symbol: {symbol}
Current price: {price}
Candles: {json.dumps(candles)}

Return ONLY JSON:
{{
 "symbol": "{symbol}",
 "direction": "long" | "short" | "flat",
 "probability": 0,
 "rr": 0.0,
 "entry": 0.0,
 "stop": 0.0,
 "tp1": 0.0,
 "summary": "short reason"
}}
"""


# ============================================================
# SCANNER
# ============================================================

async def scan_once(app):
    global last_signal_time

    if not SCAN_ENABLED:
        return

    symbols = await asyncio.to_thread(get_bybit_symbols)
    for sym in symbols:
        c15 = get_candles(sym, "15m")
        c1h = get_candles(sym, "1h")
        c4h = get_candles(sym, "4h")
        price = get_price(sym)
        if price is None:
            continue

        prompt = scan_prompt(sym, {"15m": c15, "1h": c1h, "4h": c4h}, price)
        raw = await asyncio.to_thread(run_gemini, prompt)
        data = extract_json(raw)
        if not data:
            continue

        if data["direction"] == "flat":
            continue
        if data["probability"] < MIN_PROB_SCAN:
            continue
        if data["rr"] < MIN_RR:
            continue

        key = (sym, data["direction"])
        t = last_signal_time.get(key)
        now = time.time()
        if t and now - t < SIGNAL_COOLDOWN_SECONDS:
            continue
        last_signal_time[key] = now

        msg = (
            f"🚨 *AI SIGNAL*\n"
            f"Symbol: `{sym}`\n"
            f"Direction: `{data['direction']}`\n"
            f"Probability: `{data['probability']}%`\n"
            f"RR: `{data['rr']}`\n"
            f"Entry: `{data['entry']}`\n"
            f"SL: `{data['stop']}`\n"
            f"TP1: `{data['tp1']}`\n"
            f"Reason: _{data['summary']}_"
        )
        await app.bot.send_message(OWNER_CHAT_ID, msg, parse_mode="Markdown")


async def scan_loop(app):
    await asyncio.sleep(5)
    while True:
        try:
            await scan_once(app)
        except Exception as e:
            print("SCAN LOOP ERR:", e)
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)


# ============================================================
# TG HANDLERS
# ============================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SCAN_ENABLED
    SCAN_ENABLED = True
    await update.message.reply_text("✅ Auto scanner ON.")


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SCAN_ENABLED
    SCAN_ENABLED = False
    await update.message.reply_text("⏹ Auto scanner OFF.")


async def handle_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Manual analysis not added yet.")


# ============================================================
# MAIN
# ============================================================

def main():
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(MessageHandler(filters.COMMAND, handle_pair))

    app.job_queue.run_repeating(lambda ctx: asyncio.create_task(scan_once(app)),
                                interval=SCAN_INTERVAL_SECONDS, first=5)

    print("Bot running...")
    app.run_polling()


if __name__ == "__main__":
    main()
