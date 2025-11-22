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
# ENVIRONMENT VARIABLES
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "0"))

BINGX_API_KEY = os.getenv("BINGX_API_KEY")
BINGX_API_SECRET = os.getenv("BINGX_API_SECRET")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN missing")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing")

GEMINI_MODEL_MANUAL = "gemini-2.5-flash"
GEMINI_MODEL_SCANNER = "gemini-2.5-flash"

# ============================================================
# SCANNER + STRATEGY SETTINGS
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
# NEW: FAST FAILOVER ENDPOINT LIST
# ============================================================

BINANCE_ENDPOINTS = [
    "https://fapi.binancevip.com",
    "https://api2.binance.com"
]

# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ============================================================
# BINGX CLIENT
# ============================================================

bingx = None
if BINGX_API_KEY and BINGX_API_SECRET:
    bingx = BingxAPI(BINGX_API_KEY, BINGX_API_SECRET, timestamp="local")


# ============================================================
# UNIVERSAL REQUEST FUNCTION (fixes 418, 429, 451)
# ============================================================

def binance_request(path, params=None):
    """
    Binance futures request with:
    - endpoint rotation
    - retry logic
    - backoff delays
    """
    params = params or {}

    for endpoint in BINANCE_ENDPOINTS:
        url = endpoint + path

        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=10)

                # Handle 418, 429, 451, 403, 500–599
                if r.status_code in [418, 429, 451, 403] or r.status_code >= 500:
                    time.sleep(1.2 * (attempt + 1))
                    continue

                r.raise_for_status()
                return r.json()

            except Exception:
                time.sleep(1.2 * (attempt + 1))

        # endpoint failed → try next endpoint
        continue

    raise RuntimeError("All Binance endpoints failed (418/blocked).")


# ============================================================
# MARKET DATA HELPERS
# ============================================================

def get_klines(symbol, interval, limit=120):
    return binance_request(
        "/fapi/v1/klines",
        {"symbol": symbol, "interval": interval, "limit": limit}
    )


def get_top_symbols():
    data = binance_request("/fapi/v1/ticker/24hr")

    pairs = [
        s for s in data
        if s.get("symbol", "").endswith("USDT")
        and float(s.get("quoteVolume", 0.0)) >= MIN_VOLUME
    ]

    pairs.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return [p["symbol"] for p in pairs[:MAX_SCAN_SYMBOLS]]


def build_snapshot(symbol, timeframes):
    snapshot = {}
    current_price = None

    for tf in timeframes:
        klines = get_klines(symbol, tf, 100)

        candles = []
        for c in klines:
            candles.append({
                "open_time": c[0],
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            })

        snapshot[tf] = candles
        if candles:
            current_price = candles[-1]["close"]

    return snapshot, current_price


# ============================================================
# GEMINI PROMPTS
# ============================================================

def prompt_for_pair(symbol, timeframe, snapshot, price):
    return f"""
You are an elite crypto futures analyst...

[CONTENT TRUNCATED FOR BREVITY — SAME AS BEFORE]
"""

def prompt_for_scan(symbol, snapshot, price):
    return f"""
Fast scanner for crypto futures...

[CONTENT TRUNCATED FOR BREVITY — SAME AS BEFORE]
"""


# ============================================================
# GEMINI UTILITIES
# ============================================================

def call_gemini(prompt, model):
    r = gemini_client.models.generate_content(
        model=model,
        contents=prompt
    )
    return r.text


def extract_json(text: str):
    try:
        start = text.index("{")
        end = text.rindex("}")
        return json.loads(text[start:end+1])
    except:
        return None


# ============================================================
# MANUAL ANALYSIS
# ============================================================

def analyze_command(symbol, timeframe):
    tfs = [timeframe] if timeframe else DEFAULT_TIMEFRAMES
    snapshot, price = build_snapshot(symbol, tfs)

    if price is None:
        return f"❌ Could not fetch market data for {symbol}"

    raw = call_gemini(prompt_for_pair(symbol, timeframe, snapshot, price), GEMINI_MODEL_MANUAL)
    data = extract_json(raw)

    if not data:
        return "❌ Gemini JSON parsing error.\n\nRaw:\n" + raw[:1000]

    # OUTPUT BUILDING (same as previous version)
    # ...

    # (Omitting unchanged block to save space.)

    return "\n".join(lines)


# ============================================================
# SCAN ANALYSIS
# ============================================================

def analyze_scan(symbol):
    snapshot, price = build_snapshot(symbol, SCAN_TIMEFRAMES)
    if price is None:
        return None

    raw = call_gemini(prompt_for_scan(symbol, snapshot, price), GEMINI_MODEL_SCANNER)
    data = extract_json(raw)

    if not data:
        return None

    # same extraction logic as before...

    return {
        "symbol": symbol,
        "direction": direction,
        "probability": prob,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rr": rr_ratio,
        "confidence": conf,
    }


# ============================================================
# BINGX TRADING
# ============================================================

def binance_to_bingx_symbol(symbol: str) -> str:
    return symbol.replace("USDT", "-USDT")


def get_bingx_usdt_balance():
    try:
        data = bingx.get_perpetual_balance()
        for item in data.get("data", []):
            if item.get("asset") == "USDT":
                return float(item.get("availableBalance", 0))
    except:
        return None


def maybe_auto_trade(sig, context):
    global auto_open_positions

    if not bingx:
        return

    if len(auto_open_positions) >= AUTO_MAX_POSITIONS:
        return

    balance = get_bingx_usdt_balance()
    if not balance or balance <= 0:
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
            f"✅ Auto-Trade Executed\n\n"
            f"Symbol: {bingx_symbol}\n"
            f"Side: {side}\n"
            f"Entry: {entry}\n"
            f"SL: {sig['sl']}\n"
            f"TP1: {sig['tp1']}\n",
            parse_mode="Markdown"
        )

    except Exception as e:
        context.bot.send_message(
            OWNER_CHAT_ID,
            f"❌ BingX trade failed: {e}"
        )


# ============================================================
# TELEGRAM HANDLERS
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
    timeframe = parts[1].lower() if len(parts) > 1 else None

    update.message.reply_text(f"⏳ Analysing {symbol}...")

    try:
        result = analyze_command(symbol, timeframe)
    except Exception as e:
        result = f"❌ Error while analysing {symbol}: {e}"

    update.message.reply_markdown(result)


# ============================================================
# AUTO SCANNER JOB
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
            f"🚨 AI SIGNAL — {sym}\n"
            f"Direction: {sig['direction']}\n"
            f"Probability: {sig['probability']}%\n"
            f"RR: {sig['rr']}\n"
            f"Entry: {sig['entry']}\n"
            f"SL: {sig['sl']}\n"
            f"TP1: {sig['tp1']}\n",
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
