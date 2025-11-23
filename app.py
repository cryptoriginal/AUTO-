# ============================================================
# FULL APP.PY — STABLE SCANNER + MANUAL ANALYSIS
# WITH BULLETPROOF GEMINI JSON PARSING + BINGX AUTOTRADE
# ============================================================

import os
import json
import re
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

model = genai.GenerativeModel(
    model_name=GEMINI_MODEL,
    generation_config={
        "response_mime_type": "application/json",
        "temperature": 0.2
    }
)

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
        r = requests.get(
            f"{BYBIT_ENDPOINT}/v5/market/tickers",
            params={"category": "linear"},
            timeout=10
        )
        data = r.json()
        lst = data.get("result", {}).get("list", [])
        lst = [x for x in lst if float(x["turnover24h"]) >= MIN_VOLUME]
        lst.sort(key=lambda x: float(x["turnover24h"]), reverse=True)
        return [x["symbol"] for x in lst[:25]]
    except:
        return []


def get_candles(symbol, tf):
    try:
        r = requests.get(
            f"{BYBIT_ENDPOINT}/v5/market/kline",
            params={
                "category": "linear",
                "symbol": symbol,
                "interval": INTERVAL_MAP[tf],
                "limit": 200
            },
            timeout=10
        )
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
        r = requests.get(
            f"{BYBIT_ENDPOINT}/v5/market/tickers",
            params={"category": "linear", "symbol": symbol},
            timeout=10
        )
        d = r.json()
        items = d.get("result", {}).get("list", [])
        if not items:
            return None
        return float(items[0]["lastPrice"])
    except:
        return None
# ============================================================
# BULLETPROOF GEMINI JSON EXTRACTOR
# ============================================================

def force_json(text: str):
    """
    Extract the first valid JSON object.
    This NEVER fails — last-resort fallback returns {}.
    """
    if not text:
        return {}

    # 1️⃣ Try normal JSON load directly
    try:
        return json.loads(text)
    except:
        pass

    # 2️⃣ Try extract JSON using bracket matching
    try:
        start = text.index("{")
        end = text.rindex("}")
        chunk = text[start:end+1]
        return json.loads(chunk)
    except:
        pass

    # 3️⃣ Try regex-based JSON curly capture
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass

    # 4️⃣ Last fallback → return empty JSON instead of crashing
    return {}


# ============================================================
# GEMINI PROMPTS
# ============================================================

def build_scan_prompt(symbol, candles, price):
    return {
        "symbol": symbol,
        "price": price,
        "candles": candles,
        "instruction": f"""
You are an expert crypto analyst.

Return ONLY JSON in this EXACT structure:

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

NO text outside the JSON.
"""
    }


def build_manual_prompt(symbol, snapshot, price):
    return {
        "symbol": symbol,
        "price": price,
        "snapshot": snapshot,
        "instruction": f"""
Perform a multi-timeframe technical analysis.

Return ONLY this JSON:

{{
 "symbol": "{symbol}",
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

Do NOT include anything except valid JSON.
"""
    }


# ============================================================
# GEMINI CALL
# ============================================================

def ask_gemini(prompt_dict):
    """
    Wrapper so Gemini always returns clean JSON.
    """
    try:
        resp = model.generate_content(prompt_dict)
        txt = resp.text or ""
        return force_json(txt)
    except Exception as e:
        print("Gemini error:", str(e))
        return {}


# ============================================================
# MANUAL ANALYSIS
# ============================================================

async def analyze_manual(symbol: str):
    # Timeframes used
    tfs = ["15m", "1h", "4h"]

    snapshot = {}
    price = get_price(symbol)

    if price is None:
        return f"❌ Could not fetch price for {symbol}."

    for tf in tfs:
        snapshot[tf] = get_candles(symbol, tf)

    prompt = build_manual_prompt(symbol, snapshot, price)
    data = await asyncio.to_thread(ask_gemini, prompt)

    if not data:
        return "❌ Gemini JSON error."

    direction = data.get("direction", "flat")
    summary = data.get("summary", "-")

    msg = (
        f"📊 *{symbol} Analysis*\n"
        f"Price: `{price}`\n"
        f"Direction: *{direction}*\n"
        f"Reason: _{summary}_"
    )

    entry = data.get("entry")
    stop = data.get("stop")
    tp1 = data.get("tp1")
    tp2 = data.get("tp2")

    if entry and stop and tp1:
        msg += (
            f"\n\nEntry: `{entry}`\n"
            f"SL: `{stop}`\n"
            f"TP1: `{tp1}`"
        )
        if tp2:
            msg += f"\nTP2: `{tp2}`"

    return msg
# ============================================================
# AUTOTRADE (OPTION A — ALWAYS EXECUTE)
# ============================================================

def maybe_autotrade(signal, bot):
    """
    Executes order ONLY IF:
      - symbol supported on BingX
      - <=2 open positions
    """

    if not bingx:
        return

    sym = signal["symbol"]
    direction = signal["direction"]

    if sym not in SUPPORTED_BINGX:
        print(f"{sym} not supported on BingX — skipping autotrade.")
        return

    if len(auto_open_positions) >= AUTO_MAX_POSITIONS:
        print("Max positions reached — not auto trading.")
        return

    balance = get_bingx_usdt_balance()
    if not balance:
        bot.send_message(OWNER_CHAT_ID, "⚠️ AutoTrade skipped: No USDT balance.")
        return

    entry = signal["entry"]
    if not entry or entry <= 0:
        return

    notional = balance * AUTO_LEVERAGE
    qty = round((notional / AUTO_MAX_POSITIONS) / entry, 4)

    bingx_symbol = f"{sym.replace('USDT','')}-USDT"
    side = "LONG" if direction == "long" else "SHORT"

    try:
        bingx.open_market_order(
            bingx_symbol,
            side,
            qty,
            tp=str(signal["tp1"]),
            sl=str(signal["stop"])
        )
        auto_open_positions.add(sym)

        bot.send_message(
            OWNER_CHAT_ID,
            f"✅ AutoTrade Executed\n"
            f"Symbol: {bingx_symbol}\n"
            f"Side: {side}\n"
            f"Qty: {qty}\n"
            f"Entry: {entry}\n"
            f"SL: {signal['stop']}\n"
            f"TP1: {signal['tp1']}"
        )
    except Exception as e:
        bot.send_message(OWNER_CHAT_ID, f"❌ AutoTrade Error: {e}")


# ============================================================
# SCANNER LOGIC
# ============================================================

async def analyze_signal(symbol):
    """
    Run autoscan logic on a single symbol.
    """
    price = get_price(symbol)
    if price is None:
        return None

    candles = {
        "15m": get_candles(symbol, "15m"),
        "1h": get_candles(symbol, "1h"),
        "4h": get_candles(symbol, "4h"),
    }

    prompt = build_scan_prompt(symbol, candles, price)
    data = await asyncio.to_thread(ask_gemini, prompt)

    if not data:
        return None
    if data["direction"] == "flat":
        return None
    if data.get("probability", 0) < MIN_PROB_SCAN:
        return None
    if data.get("rr", 0) < MIN_RR:
        return None

    try:
        entry = float(data["entry"])
        stop = float(data["stop"])
        tp1 = float(data["tp1"])
    except:
        return None

    return {
        "symbol": symbol,
        "direction": data["direction"],
        "probability": data["probability"],
        "entry": entry,
        "stop": stop,
        "tp1": tp1,
        "rr": data["rr"],
        "summary": data.get("summary", "")
    }


async def scan_once(app):
    if not SCAN_ENABLED:
        return

    symbols = get_bybit_symbols()
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

        warn = ""
        if sym not in SUPPORTED_BINGX:
            warn = f"ℹ️ {sym} not on BingX → no autotrade.\n\n"

        text = (
            warn +
            f"🚨 *AI SIGNAL*\n"
            f"Symbol: `{sym}`\n"
            f"Direction: `{sig['direction']}`\n"
            f"Probability: `{sig['probability']}%`\n"
            f"RR: `{sig['rr']}`\n"
            f"Entry: `{sig['entry']}`\n"
            f"SL: `{sig['stop']}`\n"
            f"TP1: `{sig['tp1']}`\n"
            f"Reason: _{sig['summary']}_"
        )

        await app.bot.send_message(OWNER_CHAT_ID, text, parse_mode="Markdown")

        # ALWAYS auto-trade (Option A)
        maybe_autotrade(sig, app.bot)


# ============================================================
# SCANNER BACKGROUND LOOP
# ============================================================

async def scanner_loop(app):
    await asyncio.sleep(5)
    while True:
        try:
            await scan_once(app)
        except Exception as e:
            print("SCAN LOOP ERROR:", e)
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)


# ============================================================
# TELEGRAM COMMAND HANDLERS
# ============================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SCAN_ENABLED
    SCAN_ENABLED = True
    await update.message.reply_text(
        "✅ Auto-scanner ON.\n"
        f"Signals every {SCAN_INTERVAL_SECONDS//60} minutes.\n"
        "Probability ≥ 80%, RR ≥ 1.9."
    )


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SCAN_ENABLED
    SCAN_ENABLED = False
    await update.message.reply_text("⛔ Auto-scanner OFF.")


async def cmd_manual(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message.text.split()
    symbol = msg[0].replace("/", "").upper()

    await update.message.reply_text(f"⏳ Analysing {symbol}...")

    result = await analyze_manual(symbol)
    await update.message.reply_markdown(result)


async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send something like:  \n`/BTCUSDT`")


# ============================================================
# MAIN
# ============================================================

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(MessageHandler(filters.Regex(r"^[A-Za-z]+USDT$"), cmd_manual))
    app.add_handler(MessageHandler(filters.COMMAND, unknown))

    app.job_queue.run_repeating(lambda ctx: asyncio.create_task(scan_once(app)),
                                interval=SCAN_INTERVAL_SECONDS,
                                first=5)

    print("BOT RUNNING…")
    app.run_polling()


if __name__ == "__main__":
    main()
# ============================================================
# PART 4 — OPTIONAL AUTO-TRADE CONFIRMATION SYSTEM
# ============================================================

pending_confirmation = None     # store pending signal
confirmation_timeout = 60       # 60 seconds to accept trade


async def ask_for_confirmation(signal, bot):
    """
    Sends /yes or /no question BEFORE opening a trade.
    """
    global pending_confirmation

    sym = signal["symbol"]
    direction = signal["direction"]

    pending_confirmation = {
        "signal": signal,
        "time": time.time()
    }

    msg = (
        f"⚠️ *Auto-Trade Confirmation Required*\n\n"
        f"Symbol: `{sym}`\n"
        f"Direction: *{direction}*\n"
        f"Entry: `{signal['entry']}`\n"
        f"SL: `{signal['stop']}`\n"
        f"TP1: `{signal['tp1']}`\n\n"
        f"💬 Reply `/yes` to execute or `/no` to cancel."
    )

    await bot.send_message(OWNER_CHAT_ID, msg, parse_mode="Markdown")


async def cmd_yes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Accept trade.
    """
    global pending_confirmation

    if not pending_confirmation:
        await update.message.reply_text("❌ No pending trade.")
        return

    sig = pending_confirmation["signal"]
    age = time.time() - pending_confirmation["time"]

    if age > confirmation_timeout:
        pending_confirmation = None
        await update.message.reply_text("⏳ Trade request expired.")
        return

    # Run autotrade
    maybe_autotrade(sig, context.bot)
    pending_confirmation = None

    await update.message.reply_text("✅ Trade executed.")


async def cmd_no(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Reject trade.
    """
    global pending_confirmation

    if not pending_confirmation:
        await update.message.reply_text("❌ No pending trade.")
        return

    pending_confirmation = None
    await update.message.reply_text("❌ Trade cancelled.")


# Replace direct autotrade call inside scan_once with:
#
#    await ask_for_confirmation(sig, app.bot)
#
# instead of:
#
#    maybe_autotrade(sig, app.bot)
#
# Example:
#
# In scan_once(), replace this:
#    maybe_autotrade(sig, app.bot)
#
# With this:
#    await ask_for_confirmation(sig, app.bot)
#
