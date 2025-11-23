# FULL APP.PY — RENDER + PTB20 + ASYNC SCANNER (NO JOBQUEUE)
# -----------------------------------------------------------

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

SCAN_INTERVAL_SECONDS = 300          # 5 minutes
SIGNAL_COOLDOWN_SECONDS = 600        # 10 minutes
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
    # Simple init – matches your installed py-bingx
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
    """Top linear USDT futures symbols by 24h turnover."""
    try:
        r = requests.get(
            f"{BYBIT_ENDPOINT}/v5/market/tickers",
            params={"category": "linear"},
            timeout=10,
        )
        data = r.json()
        lst = data.get("result", {}).get("list", []) or []
        lst = [x for x in lst if float(x.get("turnover24h", "0")) >= MIN_VOLUME]
        lst.sort(key=lambda x: float(x.get("turnover24h", "0")), reverse=True)
        return [x["symbol"] for x in lst[:20]]
    except Exception as e:
        print("get_bybit_symbols error:", e)
        return []


def get_candles(symbol, tf):
    """Fetch OHLCV candles from Bybit linear futures."""
    try:
        r = requests.get(
            f"{BYBIT_ENDPOINT}/v5/market/kline",
            params={
                "category": "linear",
                "symbol": symbol,
                "interval": INTERVAL_MAP[tf],
                "limit": 200,
            },
            timeout=10,
        )
        data = r.json()
        if data.get("retCode") != 0:
            return []
        out = []
        for c in data.get("result", {}).get("list", []) or []:
            out.append(
                {
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5]),
                }
            )
        # oldest -> newest
        return list(reversed(out))
    except Exception as e:
        print("get_candles error:", e)
        return []


def get_price(symbol):
    """Latest price from Bybit linear ticker."""
    try:
        r = requests.get(
            f"{BYBIT_ENDPOINT}/v5/market/tickers",
            params={"category": "linear", "symbol": symbol},
            timeout=10,
        )
        d = r.json()
        x = d.get("result", {}).get("list", []) or []
        if not x:
            return None
        return float(x[0].get("lastPrice"))
    except Exception as e:
        print("get_price error:", e)
        return None


# ============================================================
# GEMINI
# ============================================================

def run_gemini(prompt: str) -> str:
    try:
        resp = model.generate_content(prompt)
        return resp.text or ""
    except Exception as e:
        print("run_gemini error:", e)
        return ""


def extract_json(txt: str):
    if not txt:
        return None
    try:
        s = txt.index("{")
        e = txt.rindex("}")
        return json.loads(txt[s : e + 1])
    except Exception as e:
        print("extract_json error:", e, "text:", txt[:200])
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
    """Run one full market scan and send signals."""
    global last_signal_time

    if not SCAN_ENABLED:
        return
    if OWNER_CHAT_ID == 0:
        # no owner set, skip sending
        return

    # 1) Get universe
    symbols = await asyncio.to_thread(get_bybit_symbols)
    if not symbols:
        return

    # 2) Per symbol analysis
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

        if data.get("direction") == "flat":
            continue
        if int(data.get("probability", 0)) < MIN_PROB_SCAN:
            continue
        if float(data.get("rr", 0.0)) < MIN_RR:
            continue

        key = (sym, data["direction"])
        t = last_signal_time.get(key)
        now = time.time()
        if t and now - t < SIGNAL_COOLDOWN_SECONDS:
            # still in cooldown
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


async def scanner_loop(app):
    """Background loop that keeps scanning every SCAN_INTERVAL_SECONDS."""
    await asyncio.sleep(5)
    while True:
        try:
            await scan_once(app)
        except Exception as e:
            print("SCAN LOOP ERR:", e)
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)


async def post_init(app):
    """Hook called by PTB after application is initialized."""
    # Start background scanner as a task on PTB's event loop
    app.create_task(scanner_loop(app))


# ============================================================
# TG HANDLERS
# ============================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SCAN_ENABLED
    SCAN_ENABLED = True
    await update.message.reply_text(
        "✅ Auto scanner ON.\n"
        f"Scanning Bybit linear futures every {SCAN_INTERVAL_SECONDS//60} minutes.\n"
        f"Signals only if probability ≥ {MIN_PROB_SCAN}% and RR ≥ {MIN_RR}.\n"
        "Cooldown per pair/direction: 10 minutes."
    )


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SCAN_ENABLED
    SCAN_ENABLED = False
    await update.message.reply_text("⏹ Auto scanner OFF. Manual commands still work.")


async def handle_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Placeholder for future manual analysis
    await update.message.reply_text(
        "Manual analysis is not implemented in this minimal version yet."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)   # attach scanner loop
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(MessageHandler(filters.COMMAND, handle_pair))

    print("Bot running...")
    application.run_polling()


if __name__ == "__main__":
    main()

