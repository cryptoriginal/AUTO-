import os
import json
import time
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

gemini_model = "gemini-2.5-flash"

SCAN_ENABLED = True
SCAN_INTERVAL_SECONDS = 300        # 5 minutes
SIGNAL_COOLDOWN_SECONDS = 600      # 10 minutes
MIN_VOLUME = 50_000_000
MIN_PROB = 80
MIN_RR = 1.9

SUPPORTED = set()
last_signal = {}

DEFAULT_TF = ["5m", "15m", "1h", "4h"]


# ============================================================
# CLIENTS
# ============================================================

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

bingx = BingxAPI(BINGX_API_KEY, BINGX_API_SECRET, timestamp="local")


# ============================================================
# BINGX SYMBOL LOADER
# ============================================================

def load_bingx_symbols():
    global SUPPORTED
    try:
        data = bingx.swap_v2_get_contracts()
        lst = data.get("data", {}).get("contracts", [])
        SUPPORTED = {c["symbol"] for c in lst}
        print("Loaded BingX symbols:", SUPPORTED)
    except Exception as e:
        print("Symbol load error:", e)
        SUPPORTED = set()


# ============================================================
# MARKET DATA (ONLY BINGX)
# ============================================================

def get_candles(symbol, interval):
    try:
        result = bingx.market_get_candles(symbol, interval, 150)
        candles = result.get("data", [])
        if not candles:
            return None

        return [{
            "time": c["t"],
            "open": float(c["o"]),
            "high": float(c["h"]),
            "low": float(c["l"]),
            "close": float(c["c"]),
            "volume": float(c["v"])
        } for c in candles]
    except:
        return None


def get_live_price(symbol):
    try:
        tick = bingx.market_get_price(symbol)
        return float(tick.get("data", {}).get("price"))
    except:
        return None


def get_top_bingx_symbols():
    try:
        d = bingx.market_get_ticker_all()
        lst = d.get("data", [])
        valid = []
        for s in lst:
            if s["symbol"] not in SUPPORTED:
                continue
            if float(s.get("quoteVolume", 0)) >= MIN_VOLUME:
                valid.append(s)
        valid.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
        return [v["symbol"] for v in valid]
    except:
        return []


# ============================================================
# SNAPSHOT BUILDER
# ============================================================

def build_snapshot(symbol):
    snap = {}
    for tf in DEFAULT_TF:
        c = get_candles(symbol, tf)
        snap[tf] = c
    price = get_live_price(symbol)
    return snap, price


# ============================================================
# AI PROMPTS
# ============================================================

def prompt_analysis(symbol, snap, price):
    return f"""
Act as a world-class crypto futures analyst.

Analyze {symbol} using:
- VWAP behaviour
- Fixed range volume profile (last swing high/low)
- Support/resistance
- Liquidity sweeps
- Reversal candle patterns
- Trend strength
- Volume confirmation

Return JSON ONLY:

{{
 "direction": "long/short",
 "reason": "short explanation",
 "up": 0-100,
 "down": 0-100,
 "flat": 0-100,
 "entry": price,
 "sl": level,
 "tp": level,
 "rr": float
}}

Snapshot: {json.dumps(snap)}
Price: {price}
"""


def prompt_scan(symbol, snap, price):
    return f"""
Fast scan for {symbol}.

Return JSON ONLY with:
direction, probability, entry, sl, tp, rr.

Snapshot: {json.dumps(snap)}
Price: {price}
"""


# ============================================================
# AI HELPERS
# ============================================================

def ai(prompt):
    r = gemini_client.models.generate_content(
        model=gemini_model,
        contents=prompt
    )
    return r.text


def extract_json(text):
    try:
        a = text.index("{")
        b = text.rindex("}")
        return json.loads(text[a:b+1])
    except:
        return None


# ============================================================
# AUTO TRADE
# ============================================================

def auto_trade(sig: dict, app: ContextTypes.DEFAULT_TYPE):

    bal = get_balance()
    if bal is None or bal <= 0:
        app.bot.send_message(OWNER_CHAT_ID, "⚠️ AutoTrade fail: No balance.")
        return

    qty = round((bal * 3) / sig["entry"], 4)

    side = "LONG" if sig["direction"] == "long" else "SHORT"

    try:
        order = bingx.open_market_order(
            sig["symbol"],
            side,
            str(qty),
            tp=str(sig["tp"]),
            sl=str(sig["sl"])
        )
        app.bot.send_message(
            OWNER_CHAT_ID,
            f"✅ AutoTrade Executed\n{sig['symbol']} {side}\nQty {qty}\nSL {sig['sl']} TP {sig['tp']}"
        )
    except Exception as e:
        app.bot.send_message(OWNER_CHAT_ID, f"❌ AutoTrade error: {e}")


def get_balance():
    try:
        data = bingx.swap_v2_get_balance()
        for b in data.get("data", {}).get("balance", []):
            if b["asset"] == "USDT":
                return float(b["availableBalance"])
        return None
    except:
        return None


# ============================================================
# ANALYSIS CMD
# ============================================================

async def analyse(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().upper()
    symbol = text.replace("/", "")

    if symbol not in SUPPORTED:
        await update.message.reply_text(f"❌ {symbol} not supported on BingX.")
        return

    await update.message.reply_text(f"⏳ Analyzing {symbol}...")

    snap, price = build_snapshot(symbol)
    if not price:
        await update.message.reply_text(f"❌ Could not fetch live price.")
        return

    raw = ai(prompt_analysis(symbol, snap, price))
    data = extract_json(raw)

    if not data:
        await update.message.reply_text("❌ AI format error.")
        return

    # Skip SL/TP if probability < 80
    if max(data["up"], data["down"]) < MIN_PROB:
        await update.message.reply_text(
            f"📊 {symbol} Analysis\n"
            f"Price: {price}\n"
            f"Direction: {data['direction']}\n"
            f"Upside: {data['up']}%\n"
            f"Downside: {data['down']}%\n"
            f"Reason: {data['reason']}\n\n"
            f"⚠️ Probability < {MIN_PROB}%. No trade suggested."
        )
        return

    await update.message.reply_text(
        f"📊 {symbol} Analysis\n"
        f"Price: {price}\n"
        f"Direction: {data['direction']}\n"
        f"Upside: {data['up']}%\n"
        f"Downside: {data['down']}%\n"
        f"Reason: {data['reason']}\n\n"
        f"Entry: {data['entry']}\n"
        f"SL: {data['sl']}\n"
        f"TP: {data['tp']}\n"
        f"RR: {data['rr']}"
    )


# ============================================================
# SCANNER
# ============================================================

async def scanner(app: ContextTypes.DEFAULT_TYPE):
    if not SCAN_ENABLED:
        return

    now = datetime.now(timezone.utc)
    syms = get_top_bingx_symbols()

    for s in syms:
        # cooldown
        if s in last_signal:
            if (now - last_signal[s]).total_seconds() < SIGNAL_COOLDOWN_SECONDS:
                continue

        snap, price = build_snapshot(s)
        if not price:
            continue

        raw = ai(prompt_scan(s, snap, price))
        data = extract_json(raw)
        if not data:
            continue

        if data.get("probability", 0) < MIN_PROB:
            continue
        if data.get("rr", 0) < MIN_RR:
            continue

        last_signal[s] = now

        await app.bot.send_message(
            OWNER_CHAT_ID,
            f"🚨 AI SIGNAL\n{s}\nDir: {data['direction']}\nProb: {data['probability']}%\n"
            f"Entry {data['entry']}\nSL {data['sl']}\nTP {data['tp']}\nRR {data['rr']}"
        )

        auto_trade({
            "symbol": s,
            "direction": data["direction"],
            "entry": float(data["entry"]),
            "sl": float(data["sl"]),
            "tp": float(data["tp"])
        }, app)


# ============================================================
# COMMANDS
# ============================================================

async def cmd_start(update: Update, ctx):
    global SCAN_ENABLED
    SCAN_ENABLED = True
    await update.message.reply_text("✅ AutoScan ON (5 min scan, 10 min cooldown)")


async def cmd_stop(update: Update, ctx):
    global SCAN_ENABLED
    SCAN_ENABLED = False
    await update.message.reply_text("⛔ AutoScan OFF")


# ============================================================
# MAIN
# ============================================================

async def main():
    load_bingx_symbols()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stop", cmd_stop))

    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), analyse))

    app.job_queue.run_repeating(scanner, interval=SCAN_INTERVAL_SECONDS, first=10)

    await app.run_polling()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

