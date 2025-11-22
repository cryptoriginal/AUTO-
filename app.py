import os
import json
import time
from datetime import datetime, timezone

import requests
from google import genai
from telegram import Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    CallbackContext,
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

GEMINI_MODEL_MANUAL = "gemini-2.5-flash"
GEMINI_MODEL_SCANNER = "gemini-2.5-flash"


# ============================================================
# SCANNER SETTINGS
# ============================================================

SCAN_ENABLED = True
SCAN_INTERVAL_SECONDS = 300

MIN_VOLUME = 50_000_000
MAX_SCAN_SYMBOLS = 25

MIN_PROB_MANUAL = 75
MIN_PROB_SCAN = 85
MIN_RR = 2.1

DEFAULT_TIMEFRAMES = [
    "5m", "15m", "30m", "1h", "2h", "4h",
    "6h", "12h", "1d", "1w"
]

SCAN_TIMEFRAMES = ["15m", "1h", "4h"]

AUTO_MAX_POSITIONS = 2
AUTO_LEVERAGE = 3.0

last_signal_time = {}
auto_open_positions = set()


# ============================================================
# CLIENTS
# ============================================================

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

bingx = None
if BINGX_API_KEY and BINGX_API_SECRET:
    bingx = BingxAPI(BINGX_API_KEY, BINGX_API_SECRET, timestamp="local")


# ============================================================
# MARKET DATA ENDPOINTS (TRIPLE FALLBACK)
# ============================================================

BINANCE_ENDPOINTS = [
    "https://fapi.binancevip.com",
    "https://api2.binance.com"
]

OKX_ENDPOINT = "https://www.okx.com"


# ============================================================
# UNIVERSAL REQUEST FUNCTIONS
# ============================================================

def fetch_binance(path, params=None):
    """Fallback #3: Binance futures with retry + backoff."""
    params = params or {}

    for endpoint in BINANCE_ENDPOINTS:
        url = endpoint + path

        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=10)

                if r.status_code in [418, 429, 451, 403] or r.status_code >= 500:
                    time.sleep(1.2 * (attempt + 1))
                    continue

                r.raise_for_status()
                return r.json()
            except:
                time.sleep(1.2 * (attempt + 1))

    raise RuntimeError("Binance failed in all attempts.")


def fetch_okx(symbol, interval):
    """Fallback #2: OKX Candles"""
    try:
        url = f"{OKX_ENDPOINT}/api/v5/market/candles"
        r = requests.get(url, params={"instId": f"{symbol}-USDT-SWAP", "bar": interval}, timeout=10)
        r.raise_for_status()
        raw = r.json()
        candles = []
        for c in raw.get("data", []):
            candles.append({
                "open_time": c[0],
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            })
        return candles
    except:
        return None


def fetch_bingx(symbol, interval):
    """Primary: BingX USDT-M Futures candles"""
    try:
        data = bingx.market_get_candles(symbol, interval, 100)
        candles = []
        for c in data.get("data", []):
            candles.append({
                "open_time": c["t"],
                "open": float(c["o"]),
                "high": float(c["h"]),
                "low": float(c["l"]),
                "close": float(c["c"]),
                "volume": float(c["v"]),
            })
        return candles
    except:
        return None


# Wrapper for triple fallback
def get_klines(symbol, interval):
    # 1️⃣ Try BingX
    c = fetch_bingx(symbol, interval)
    if c and len(c) > 0:
        return c

    # 2️⃣ Try OKX
    c = fetch_okx(symbol, interval)
    if c and len(c) > 0:
        return c

    # 3️⃣ Try Binance
    raw = fetch_binance("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": 100})
    candles = []
    for c in raw:
        candles.append({
            "open_time": c[0],
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]),
        })
    return candles


def get_top_symbols():
    """Get top symbols by 24h volume from Binance (safe to use occasionally)."""
    try:
        data = fetch_binance("/fapi/v1/ticker/24hr")
    except:
        return []

    pairs = [
        s for s in data
        if s.get("symbol", "").endswith("USDT")
        and float(s.get("quoteVolume", 0)) >= MIN_VOLUME
    ]
    pairs.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return [p["symbol"] for p in pairs[:MAX_SCAN_SYMBOLS]]


# ============================================================
# SNAPSHOT BUILDER
# ============================================================

def build_snapshot(symbol, timeframes):
    snapshot = {}
    current_price = None

    for tf in timeframes:
        klines = get_klines(symbol, tf)
        snapshot[tf] = klines
        if klines:
            current_price = klines[-1]["close"]

    return snapshot, current_price


# ============================================================
# GEMINI PROMPTS
# (content omitted to keep message short — identical logic)
# ============================================================

def prompt_for_pair(symbol, timeframe, snapshot, price):
    return f"""
You are an expert analyst...
(keep your previous prompt text here)
"""

def prompt_for_scan(symbol, snapshot, price):
    return f"""
You are a fast crypto scanner...
(keep your previous prompt text here)
"""


# ============================================================
# GEMINI HELPERS
# ============================================================

def call_gemini(prompt, model):
    r = gemini_client.models.generate_content(model=model, contents=prompt)
    return r.text


def extract_json(text):
    try:
        start = text.index("{")
        end = text.rindex("}")
        return json.loads(text[start:end+1])
    except:
        return None


# ============================================================
# BINGX BALANCE (FIXED FOR py-bingx==0.4)
# ============================================================

def get_bingx_usdt_balance():
    """Correct balance endpoint for USDS-M Perp on py-bingx==0.4"""
    try:
        data = bingx.swap_v2_get_balance()
        balances = data.get("data", {}).get("balance", [])
        for b in balances:
            if b.get("asset") == "USDT":
                return float(b.get("availableBalance", 0))
        return None
    except Exception as e:
        print("BalanceError:", e)
        return None


# ============================================================
# SYMBOL CONVERSION
# ============================================================

def binance_to_bingx_symbol(symbol):
    return symbol.replace("USDT", "-USDT")


# ============================================================
# AUTO-TRADE EXECUTION
# ============================================================

def maybe_auto_trade(sig, context):
    global auto_open_positions

    if not bingx:
        return
    if len(auto_open_positions) >= AUTO_MAX_POSITIONS:
        return

    balance = get_bingx_usdt_balance()
    if not balance or balance <= 0:
        context.bot.send_message(OWNER_CHAT_ID, "⚠️ Auto-trade skipped: balance unavailable.")
        return

    entry = sig["entry"]
    if entry <= 0:
        return

    total_notional = balance * AUTO_LEVERAGE
    qty = (total_notional / AUTO_MAX_POSITIONS) / entry

    bingx_symbol = binance_to_bingx_symbol(sig["symbol"])
    side = "LONG" if sig["direction"] == "long" else "SHORT"

    try:
        order = bingx.open_market_order(
            bingx_symbol,
            side,
            f"{qty:.6f}",
            tp=str(sig["tp1"]),
            sl=str(sig["sl"])
        )

        auto_open_positions.add(sig["symbol"])

        context.bot.send_message(
            OWNER_CHAT_ID,
            f"✅ AutoTrade Executed\n"
            f"Symbol: {bingx_symbol}\nSide: {side}\nQty: {qty:.6f}\n"
            f"Entry: {entry}\nSL: {sig['sl']}\nTP1: {sig['tp1']}"
        )

    except Exception as e:
        context.bot.send_message(OWNER_CHAT_ID, f"❌ AutoTrade Error: {e}")


# ============================================================
# ANALYSIS HANDLERS
# ============================================================

def analyze_command(symbol, timeframe):
    tfs = [timeframe] if timeframe else DEFAULT_TIMEFRAMES
    snapshot, price = build_snapshot(symbol, tfs)

    if price is None:
        return f"❌ Could not fetch candles for {symbol}"

    raw = call_gemini(prompt_for_pair(symbol, timeframe, snapshot, price), GEMINI_MODEL_MANUAL)
    data = extract_json(raw)

    if not data:
        return "❌ JSON parsing error.\n" + raw[:800]

    result = [
        f"📊 *{symbol} Analysis*",
        f"Price: `{price}`",
        f"Direction: *{data.get('direction')}*",
        f"Upside: `{data.get('upside_probability')}%`",
        f"Downside: `{data.get('downside_probability')}%`",
        f"Flat: `{data.get('flat_probability')}%`",
    ]

    if data.get("entry"):
        result.append("")
        result.append(f"Entry: `{data['entry']}`")
        result.append(f"SL: `{data['sl']}`")
        result.append(f"TP1: `{data['tp1']}`")
        result.append(f"TP2: `{data.get('tp2')}`")

    return "\n".join(result)


def analyze_scan(symbol):
    snapshot, price = build_snapshot(symbol, SCAN_TIMEFRAMES)

    if price is None:
        return None

    raw = call_gemini(prompt_for_scan(symbol, snapshot, price), GEMINI_MODEL_SCANNER)
    data = extract_json(raw)

    if not data:
        return None

    prob = int(data.get("probability", 0))
    rr = float(data.get("rr", 0))

    if prob < MIN_PROB_SCAN or rr < MIN_RR:
        return None

    return {
        "symbol": symbol,
        "direction": data["direction"],
        "probability": prob,
        "entry": float(data["entry"]),
        "sl": float(data["sl"]),
        "tp1": float(data["tp1"]),
        "tp2": float(data.get("tp2", data["tp1"])),
        "rr": rr,
    }


# ============================================================
# TELEGRAM COMMAND HANDLERS
# ============================================================

def start(update, context):
    global SCAN_ENABLED
    SCAN_ENABLED = True
    update.message.reply_text("✅ Auto Scanner ON")

def stop(update, context):
    global SCAN_ENABLED
    SCAN_ENABLED = False
    update.message.reply_text("⏹ Auto Scanner OFF")

def handle_pair(update, context):
    msg = update.message.text.strip()
    parts = msg.split()

    symbol = parts[0].replace("/", "").upper()
    timeframe = parts[1] if len(parts) > 1 else None

    update.message.reply_text(f"⏳ Analysing {symbol}...")

    try:
        result = analyze_command(symbol, timeframe)
    except Exception as e:
        result = f"❌ Error analysing {symbol}: {e}"

    update.message.reply_markdown(result)


# ============================================================
# SCANNER JOB
# ============================================================

def scanner_job(context):
    if not SCAN_ENABLED:
        return
    if OWNER_CHAT_ID == 0:
        return

    try:
        symbols = get_top_symbols()
    except:
        return

    now = datetime.now(timezone.utc)

    for sym in symbols:
        try:
            sig = analyze_scan(sym)
        except:
            continue

        if not sig:
            continue

        key = (sym, sig["direction"])
        last = last_signal_time.get(key)

        if last and (now - last).total_seconds() < 1800:
            continue

        last_signal_time[key] = now

        context.bot.send_message(
            OWNER_CHAT_ID,
            f"🚨 *AI SIGNAL*\n"
            f"Symbol: `{sym}`\n"
            f"Direction: `{sig['direction']}`\n"
            f"Probability: `{sig['probability']}%`\n"
            f"RR: `{sig['rr']}`\n"
            f"Entry: `{sig['entry']}`\n"
            f"SL: `{sig['sl']}`\n"
            f"TP1: `{sig['tp1']}`",
            parse_mode="Markdown"
        )

        maybe_auto_trade(sig, context)


# ============================================================
# MAIN
# ============================================================

def main():
    updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("stop", stop))
    dp.add_handler(MessageHandler(Filters.command, handle_pair))

    jq = updater.job_queue
    jq.run_repeating(scanner_job, interval=SCAN_INTERVAL_SECONDS, first=10)

    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
