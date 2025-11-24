# ============================================================
# FULL CLEAN APP.PY — GEMINI-1.5-PRO + BYBIT + BINGX AUTOTRADE
# Ultra-stable JSON, Autoscan, Manual Analysis, Render-safe
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
    ApplicationBuilder, CommandHandler,
    MessageHandler, ContextTypes, filters
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

# Best model (you requested Option A)
GEMINI_MODEL = "gemini-1.5-pro"


# ============================================================
# SETTINGS
# ============================================================

SCAN_ENABLED = True
SCAN_INTERVAL = 300           # 5 min
MIN_VOLUME = 50_000_000
MIN_PROB = 80
MIN_RR = 1.9
SIGNAL_COOLDOWN = 600         # 10 min

AUTO_LEVERAGE = 3
AUTO_MAX_POS = 2

SUPPORTED_BINGX = set()
last_signal_time = {}
auto_positions = set()


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

BYBIT = "https://api.bybit.com"

INTERVAL_MAP = {"15m": "15", "1h": "60", "4h": "240"}


def bybit_symbols():
    """Return top high-volume USDT perpetuals."""
    try:
        r = requests.get(
            f"{BYBIT}/v5/market/tickers",
            params={"category": "linear"},
            timeout=10
        )
        lst = r.json().get("result", {}).get("list", [])
        lst = [
            x for x in lst
            if x["symbol"].endswith("USDT")
            and float(x["turnover24h"]) >= MIN_VOLUME
        ]
        lst.sort(key=lambda x: float(x["turnover24h"]), reverse=True)
        return [x["symbol"] for x in lst[:25]]
    except:
        return []


def get_price(symbol):
    try:
        r = requests.get(
            f"{BYBIT}/v5/market/tickers",
            params={"category": "linear", "symbol": symbol},
            timeout=10
        )
        lst = r.json().get("result", {}).get("list", [])
        if not lst:
            return None
        return float(lst[0]["lastPrice"])
    except:
        return None


def get_candles(symbol, tf):
    try:
        r = requests.get(
            f"{BYBIT}/v5/market/kline",
            params={
                "category": "linear",
                "symbol": symbol,
                "interval": INTERVAL_MAP[tf],
                "limit": 200,
            },
            timeout=10
        )
        if r.json().get("retCode") != 0:
            return []

        out = []
        for c in r.json()["result"]["list"]:
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


# ============================================================
# BINGX
# ============================================================

def load_bingx_symbols():
    """Loads all futures pairs."""
    global SUPPORTED_BINGX
    if not bingx:
        SUPPORTED_BINGX = set()
        return

    try:
        data = bingx.get_all_contracts()
        if isinstance(data, dict):
            items = (
                data.get("data", {}).get("contracts")
                or data.get("contracts")
                or []
            )
        else:
            items = data or []

        symbols = set()
        for c in items:
            sym = c.get("symbol") if isinstance(c, dict) else str(c)
            if not sym:
                continue
            if sym.endswith("-USDT"):
                symbols.add(sym.replace("-USDT", "USDT"))
            elif sym.endswith("USDT"):
                symbols.add(sym)
        SUPPORTED_BINGX = symbols
        print("[BINGX] loaded:", len(symbols))
    except Exception as e:
        print("load_bingx_symbols:", e)
        SUPPORTED_BINGX = set()


def bingx_balance():
    """Returns available USDT futures balance."""
    try:
        info = bingx.get_perpetual_balance()
        bal = (info.get("data") or {}).get("balance") or {}
        x = bal.get("availableMargin") or bal.get("balance")
        return float(x) if x else None
    except:
        return None


# ============================================================
# GEMINI JSON EXTRACTION
# ============================================================

def clean_json(text):
    """Extract clean JSON from Gemini output."""
    if not text:
        return {}

    # remove markdown fencing
    text = re.sub(r"```(json)?", "", text).replace("```", "").strip()

    # direct parse
    try:
        return json.loads(text)
    except:
        pass

    # find {...}
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except:
        pass

    return {}


def ask_gemini(prompt: str):
    """Call Gemini and return JSON."""
    try:
        out = model.generate_content(prompt)
        return clean_json(out.text or "")
    except Exception as e:
        print("Gemini error:", e)
        return {}


# ============================================================
# PROMPTS
# ============================================================

def manual_prompt(symbol, snapshot, price):
    return f"""
Return ONLY VALID JSON:

{{
 "direction": "long" | "short" | "flat",
 "summary": "short reason",
 "upside": 0,
 "downside": 0,
 "flat": 0,
 "entry": null | 0.0,
 "stop": null | 0.0,
 "tp1": null | 0.0,
 "tp2": null | 0.0
}}

Analyse {symbol} at price {price}.
Use snapshot: {json.dumps(snapshot)}

Rules:
- If probability < 80%, entry/stop/tp1/tp2 = null.
- Base levels only on real key levels (S/R, swing highs, VWAP, volume clusters).
"""


def scan_prompt(symbol, candles, price):
    return f"""
Return ONLY VALID JSON:

{{
 "direction": "long" | "short" | "flat",
 "probability": 0,
 "rr": 0.0,
 "entry": 0.0,
 "stop": 0.0,
 "tp1": 0.0,
 "summary": "short reason"
}}

Analyse {symbol} for a high-probability setup.
Candles = {json.dumps(candles)}
Price = {price}

Only output long/short if:
- probability ≥ 80
- rr ≥ 1.9
Otherwise return direction="flat".
"""


# ============================================================
# MANUAL ANALYSIS (/btcusdt)
# ============================================================

async def analyze_manual(symbol: str):
    price = get_price(symbol)
    if price is None:
        return f"❌ Could not fetch price for {symbol}"

    snapshot = {
        "15m": get_candles(symbol, "15m"),
        "1h": get_candles(symbol, "1h"),
        "4h": get_candles(symbol, "4h"),
    }

    data = await asyncio.to_thread(
        ask_gemini, manual_prompt(symbol, snapshot, price)
    )

    if not data:
        return "❌ AI could not generate analysis. Try again."

    direction = data.get("direction", "flat")
    summary = data.get("summary", "-")

    msg = [
        f"📊 *{symbol} Analysis*",
        f"Price: `{price}`",
        f"Direction: *{direction}*",
        f"Reason: _{summary}_",
        "",
        f"Upside: `{data.get('upside', 0)}%`",
        f"Downside: `{data.get('downside', 0)}%`",
        f"Flat: `{data.get('flat', 0)}%`",
    ]

    if data.get("entry") and data.get("stop") and data.get("tp1"):
        msg += [
            "",
            f"Entry: `{data['entry']}`",
            f"SL: `{data['stop']}`",
            f"TP1: `{data['tp1']}`",
        ]
        if data.get("tp2"):
            msg.append(f"TP2: `{data['tp2']}`")

    return "\n".join(msg)


# ============================================================
# AUTOTRADE
# ============================================================

def autotrade(signal, bot):
    if not bingx:
        return

    sym = signal["symbol"]
    if sym not in SUPPORTED_BINGX:
        return

    if len(auto_positions) >= AUTO_MAX_POS:
        return

    bal = bingx_balance()
    if not bal or bal <= 0:
        return

    entry = signal["entry"]
    qty = round((bal * AUTO_LEVERAGE / AUTO_MAX_POS) / entry, 4)

    bingx_sym = f"{sym.replace('USDT', '')}-USDT"
    side = "LONG" if signal["direction"] == "long" else "SHORT"

    try:
        bingx.open_market_order(
            bingx_sym, side, qty,
            tp=str(signal["tp1"]),
            sl=str(signal["stop"])
        )
        auto_positions.add(sym)

        bot.send_message(
            OWNER_CHAT_ID,
            f"✅ AutoTrade {bingx_sym} {side}\nQty: {qty}"
        )
    except Exception as e:
        bot.send_message(OWNER_CHAT_ID, f"❌ AutoTrade error: {e}")


# ============================================================
# AUTOSCAN
# ============================================================

async def analyze_signal(symbol):
    price = get_price(symbol)
    if price is None:
        return None

    candles = {
        "15m": get_candles(symbol, "15m"),
        "1h": get_candles(symbol, "1h"),
        "4h": get_candles(symbol, "4h"),
    }

    data = await asyncio.to_thread(
        ask_gemini, scan_prompt(symbol, candles, price)
    )

    if not data:
        return None
    if data.get("direction") == "flat":
        return None
    if int(data.get("probability", 0)) < MIN_PROB:
        return None
    if float(data.get("rr", 0)) < MIN_RR:
        return None

    return {
        "symbol": symbol,
        "direction": data["direction"],
        "probability": int(data["probability"]),
        "entry": float(data["entry"]),
        "stop": float(data["stop"]),
        "tp1": float(data["tp1"]),
        "rr": float(data["rr"]),
        "summary": data.get("summary", "")
    }


async def scan_once(app):
    if not SCAN_ENABLED:
        return

    syms = await asyncio.to_thread(bybit_symbols)
    now = time.time()

    for sym in syms:
        sig = await analyze_signal(sym)
        if not sig:
            continue

        key = (sym, sig["direction"])
        if key in last_signal_time and now - last_signal_time[key] < SIGNAL_COOLDOWN:
            continue

        last_signal_time[key] = now

        text = (
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

        autotrade(sig, app.bot)


async def scanner_loop(app):
    await asyncio.sleep(5)
    while True:
        try:
            await scan_once(app)
        except Exception as e:
            print("SCAN ERROR:", e)
        await asyncio.sleep(SCAN_INTERVAL)


# ============================================================
# TELEGRAM HANDLERS
# ============================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SCAN_ENABLED
    SCAN_ENABLED = True
    await update.message.reply_text("✅ Auto-scanner ON.")


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SCAN_ENABLED
    SCAN_ENABLED = False
    await update.message.reply_text("⏹ Auto-scanner OFF.")


async def handle_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles /btcusdt style commands."""
    msg = update.message.text.strip()
    symbol = msg.replace("/", "").split()[0].upper()

    if not symbol.endswith("USDT"):
        await update.message.reply_text("Use `/btcusdt` format")
        return

    await update.message.reply_text(f"⏳ Analysing {symbol}...")
    result = await analyze_manual(symbol)
    await update.message.reply_markdown(result)


# ============================================================
# MAIN
# ============================================================

async def post_init(app):
    await asyncio.to_thread(load_bingx_symbols)
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
        MessageHandler(filters.COMMAND & ~filters.Regex(r"^/(start|stop)$"), handle_pair)
    )

    print("Bot running...")
    app.run_polling()


if __name__ == "__main__":
    main()

