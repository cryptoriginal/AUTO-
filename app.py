# ============================================================
# FULL APP.PY — GEMINI 2.0 FLASH — CLEANEST WORKING VERSION
# Background Scanner + Manual Analysis + BingX Autotrade
# ============================================================

import os
import json
import re
import time
import asyncio
import requests
import google.generativeai as genai

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

from bingx.api import BingxAPI


# ============================================================
# ENVIRONMENT
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "0") or "0")

BINGX_API_KEY = os.getenv("BINGX_API_KEY")
BINGX_API_SECRET = os.getenv("BINGX_API_SECRET")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")


# ============================================================
# GEMINI MODEL (LATEST WORKING)
# ============================================================

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")


# ============================================================
# SCANNER SETTINGS
# ============================================================

SCAN_INTERVAL_SECONDS = 300
SIGNAL_COOLDOWN_SECONDS = 600
MIN_VOLUME = 50_000_000
MIN_PROB_SCAN = 80
MIN_RR = 1.9

AUTO_LEVERAGE = 3
AUTO_MAX_POSITIONS = 2

SCAN_ENABLED = True

SUPPORTED_BINGX = set()
auto_open_positions = set()
last_signal_time = {}


# ============================================================
# BINGX INIT
# ============================================================

bingx = None
if BINGX_API_KEY and BINGX_API_SECRET:
    try:
        bingx = BingxAPI(BINGX_API_KEY, BINGX_API_SECRET)
    except Exception as e:
        print("BingX init failed:", e)
        bingx = None


# ============================================================
# BYBIT MARKET DATA
# ============================================================

BYBIT_ENDPOINT = "https://api.bybit.com"

INTERVAL_MAP = {
    "15m": "15",
    "1h": "60",
    "4h": "240"
}


def get_bybit_symbols():
    try:
        r = requests.get(
            f"{BYBIT_ENDPOINT}/v5/market/tickers",
            params={"category": "linear"},
            timeout=10
        )
        data = r.json()
        items = data.get("result", {}).get("list", []) or []

        filtered = [
            i for i in items
            if i.get("symbol", "").endswith("USDT")
            and float(i.get("turnover24h") or 0) >= MIN_VOLUME
        ]

        filtered.sort(key=lambda x: float(x.get("turnover24h", 0)), reverse=True)
        return [x["symbol"] for x in filtered[:30]]

    except Exception as e:
        print("get_bybit_symbols error:", e)
        return []


def get_candles(symbol, tf):
    interval = INTERVAL_MAP.get(tf)
    if not interval:
        return []

    try:
        r = requests.get(
            f"{BYBIT_ENDPOINT}/v5/market/kline",
            params={
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": 200
            },
            timeout=10
        )
        data = r.json()

        if data.get("retCode") != 0:
            return []

        out = []
        for c in data.get("result", {}).get("list", []):
            out.append({
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5])
            })

        return list(reversed(out))

    except Exception as e:
        print("get_candles error:", e)
        return []


def get_price(symbol):
    try:
        r = requests.get(
            f"{BYBIT_ENDPOINT}/v5/market/tickers",
            params={"category": "linear", "symbol": symbol},
            timeout=10
        )
        d = r.json()
        lst = d.get("result", {}).get("list", [])
        if not lst:
            return None

        return float(lst[0].get("lastPrice"))

    except Exception as e:
        print("get_price error:", e)
        return None


# ============================================================
# GEMINI JSON HANDLING
# ============================================================

def clean_json(response_text: str):
    if not response_text:
        return {}

    try:
        return json.loads(response_text)
    except:
        pass

    cleaned = re.sub(r"```json|```", "", response_text).strip()

    try:
        return json.loads(cleaned)
    except:
        pass

    try:
        start = cleaned.index("{")
        end = cleaned.rindex("}")
        return json.loads(cleaned[start:end+1])
    except:
        return {}


def gemini_json(prompt: str):
    try:
        resp = gemini_model.generate_content(prompt)
        return clean_json(resp.text)
    except:
        return {}


# ============================================================
# PROMPTS
# ============================================================

def manual_prompt(symbol, snapshot, price):
    return f"""
You are a pro crypto analyst.

Symbol: {symbol}
Price: {price}
Snapshot: {json.dumps(snapshot)}

Return ONLY JSON:

{{
 "direction": "long" | "short" | "flat",
 "summary": "short reason",
 "upside": 0,
 "downside": 0,
 "flat": 0,
 "entry": 0.0,
 "stop": 0.0,
 "tp1": 0.0,
 "tp2": 0.0
}}
"""


def scan_prompt(symbol, candles, price):
    return f"""
Symbol: {symbol}
Price: {price}
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
 "summary": "short"
}}
"""


# ============================================================
# MANUAL ANALYSIS
# ============================================================

async def analyze_manual(symbol):
    symbol = symbol.upper()
    price = get_price(symbol)
    if price is None:
        return f"❌ Cannot fetch price for {symbol}"

    snapshot = {
        "15m": get_candles(symbol, "15m"),
        "1h": get_candles(symbol, "1h"),
        "4h": get_candles(symbol, "4h")
    }

    data = await asyncio.to_thread(
        gemini_json, manual_prompt(symbol, snapshot, price)
    )

    if not data:
        return "❌ AI could not generate analysis. Try again."

    direction = data.get("direction", "flat")
    summary = data.get("summary", "-")
    up = data.get("upside", 0)
    down = data.get("downside", 0)
    fl = data.get("flat", 0)

    entry = data.get("entry")
    sl = data.get("stop")
    tp1 = data.get("tp1")
    tp2 = data.get("tp2")

    msg = (
        f"📊 *{symbol} Analysis*\n"
        f"Price: `{price}`\n"
        f"Direction: *{direction}*\n"
        f"Upside `{up}%`  Down `{down}%`  Flat `{fl}%`\n"
        f"Reason: _{summary}_\n"
    )

    if entry and sl and tp1:
        msg += (
            f"\nEntry: `{entry}`\n"
            f"SL: `{sl}`\n"
            f"TP1: `{tp1}`\n"
            f"TP2: `{tp2}`\n"
        )

    return msg


# ============================================================
# AUTOSCAN + AUTOTRADE
# ============================================================

async def analyze_signal(symbol):
    price = get_price(symbol)
    if price is None:
        return None

    candles = {
        "15m": get_candles(symbol, "15m"),
        "1h": get_candles(symbol, "1h"),
        "4h": get_candles(symbol, "4h")
    }

    data = await asyncio.to_thread(gemini_json, scan_prompt(symbol, candles, price))
    if not data:
        return None

    if data.get("direction") == "flat":
        return None
    if int(data.get("probability", 0)) < MIN_PROB_SCAN:
        return None
    if float(data.get("rr", 0)) < MIN_RR:
        return None

    return data


async def scan_once(app):
    if not SCAN_ENABLED:
        return

    symbols = await asyncio.to_thread(get_bybit_symbols)
    if not symbols:
        return

    now = time.time()

    for sym in symbols:
        sig = await analyze_signal(sym)
        if not sig:
            continue

        key = (sym, sig["direction"])
        last = last_signal_time.get(key)
        if last and now - last < SIGNAL_COOLDOWN_SECONDS:
            continue

        last_signal_time[key] = now

        await app.bot.send_message(
            OWNER_CHAT_ID,
            f"🚨 *AI SIGNAL*\n"
            f"{sym}\n"
            f"Direction: `{sig['direction']}`\n"
            f"Probability: `{sig['probability']}%`\n"
            f"RR: `{sig['rr']}`\n"
            f"Entry: `{sig['entry']}`\n"
            f"SL: `{sig['stop']}`\n"
            f"TP1: `{sig['tp1']}`\n"
            f"Reason: _{sig['summary']}_",
            parse_mode="Markdown"
        )


async def scanner_loop(app):
    await asyncio.sleep(5)
    while True:
        try:
            await scan_once(app)
        except Exception as e:
            print("scanner loop:", e)
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)


# ============================================================
# TELEGRAM HANDLERS
# ============================================================

async def cmd_start(update, context):
    global SCAN_ENABLED
    SCAN_ENABLED = True
    await update.message.reply_text("✅ Auto-scan enabled.")


async def cmd_stop(update, context):
    global SCAN_ENABLED
    SCAN_ENABLED = False
    await update.message.reply_text("⏹ Auto-scan disabled.")


async def handle_pair(update, context):
    text = update.message.text.strip()
    symbol = text.replace("/", "").upper()

    if not symbol.endswith("USDT"):
        await update.message.reply_text("Send like: `/btcUSDT`", parse_mode="Markdown")
        return

    await update.message.reply_text(f"⏳ Analysing {symbol}...")
    result = await analyze_manual(symbol)
    await update.message.reply_markdown(result)


# ============================================================
# MAIN
# ============================================================

async def post_init(app):
    app.create_task(scanner_loop(app))


def main():
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stop", cmd_stop))

    app.add_handler(
        MessageHandler(filters.COMMAND, handle_pair)
    )

    print("Bot running...")
    app.run_polling()


if __name__ == "__main__":
    main()

