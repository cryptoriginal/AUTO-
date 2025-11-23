import os
import json
import time
import requests
from datetime import datetime, timezone

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


# =============================
# ENVIRONMENT VARIABLES
# =============================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "0"))

BINGX_API_KEY = os.getenv("BINGX_API_KEY")
BINGX_API_SECRET = os.getenv("BINGX_API_SECRET")

if not TELEGRAM_BOT_TOKEN: raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
if not GEMINI_API_KEY: raise RuntimeError("Missing GEMINI_API_KEY")


# =============================
# CONSTANTS
# =============================
SCAN_ENABLED = True
SCAN_INTERVAL = 300              # 5 minutes
SIGNAL_COOLDOWN_SECONDS = 600    # 10 minutes
MIN_VOLUME = 50_000_000
MIN_PROB_MANUAL = 80
MIN_PROB_SCAN = 80
MIN_RR = 1.9

DEFAULT_TFS = ["5m","15m","30m","1h","2h","4h","6h","12h","1d","1w"]
SCAN_TFS = ["15m","1h","4h"]

AUTO_MAX_POS = 2
AUTO_LEVERAGE = 3.0

last_signal_time = {}
auto_positions = set()
SUPPORTED_BINGX = set()

# =============================
# CLIENTS
# =============================
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

bingx = None
if BINGX_API_KEY and BINGX_API_SECRET:
    bingx = BingxAPI(BINGX_API_KEY, BINGX_API_SECRET, timestamp="local")


# =============================
# LOAD SUPPORTED BINGX SYMBOLS
# =============================
def load_supported_bingx_symbols():
    global SUPPORTED_BINGX
    try:
        data = bingx.swap_v2_get_contracts()
        lst = data.get("data", {}).get("contracts", [])
        SUPPORTED_BINGX = {c["symbol"].replace("-USDT", "USDT") for c in lst}
        print("Loaded BingX symbols:", len(SUPPORTED_BINGX))
    except:
        SUPPORTED_BINGX = set()


# =============================
# MARKET DATA (3-Layer Fallback)
# =============================
def fetch_bingx(symbol, tf):
    try:
        data = bingx.market_get_candles(symbol, tf, 120)
        return [{
            "open_time": c["t"],
            "open": float(c["o"]),
            "high": float(c["h"]),
            "low": float(c["l"]),
            "close": float(c["c"]),
            "volume": float(c["v"])
        } for c in data.get("data", [])]
    except:
        return None


def fetch_okx(symbol, tf):
    try:
        url = "https://www.okx.com/api/v5/market/candles"
        r = requests.get(url, params={"instId": f"{symbol}-USDT-SWAP","bar": tf}, timeout=8)
        r.raise_for_status()
        raw = r.json()

        return [{
            "open_time": c[0],
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5])
        } for c in raw.get("data", [])]
    except:
        return None


def fetch_binance(symbol, tf):
    urls = [
        "https://fapi.binancevip.com/fapi/v1/klines",
        "https://api2.binance.com/fapi/v1/klines"
    ]

    for url in urls:
        try:
            r = requests.get(url, params={"symbol":symbol,"interval":tf,"limit":120}, timeout=8)
            if r.status_code in [418,429,451,403] or r.status_code >= 500:
                continue

            r.raise_for_status()
            raw = r.json()

            return [{
                "open_time": c[0],
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5])
            } for c in raw]
        except:
            continue

    return None


def get_klines(symbol, tf):
    c = fetch_bingx(symbol, tf)
    if c: return c

    c = fetch_okx(symbol, tf)
    if c: return c

    return fetch_binance(symbol, tf)


# =============================
# TOP SYMBOL SCAN (Binance)
# =============================
def get_top_symbols():
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=8)
        r.raise_for_status()
        data = r.json()
    except:
        return []

    pairs = [
        s for s in data
        if s["symbol"].endswith("USDT")
        and float(s["quoteVolume"]) >= MIN_VOLUME
    ]

    pairs.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return [p["symbol"] for p in pairs[:25]]


# =============================
# SNAPSHOT BUILDER
# =============================
def build_snapshot(symbol, tfs):
    snap = {}
    price = None
    for tf in tfs:
        k = get_klines(symbol, tf)
        snap[tf] = k
        if k:
            price = k[-1]["close"]
    return snap, price


# =============================
# GEMINI PROMPTS
# =============================
def prompt_manual(symbol, timeframe, snapshot, price):
    return f"""
You are a world-class crypto futures analyst.

Use:
- VWAP importance
- Fixed Range Volume Profile (recent swing high–low)
- Support/Resistance
- Reversal candle patterns (hammer, doji, engulfing)
- Trend structure
- Breakout/bounce confirmation

Return ONLY JSON:
{{
 "direction": "long/short",
 "summary": "<1 sentence reason>",
 "upside_probability": %,
 "downside_probability": %,
 "flat_probability": %,
 "entry": number,
 "sl": number,
 "tp1": number,
 "tp2": number,
 "rr": number
}}

Symbol: {symbol}
Timeframe: {timeframe}
Price: {price}
Snapshot: {json.dumps(snapshot)}
"""


def prompt_scan(symbol, snapshot, price):
    return f"""
Act as a high-speed crypto scanner using:
VWAP + Fixed Range Volume Profile + candle patterns.

Return ONLY JSON with:
{{
 "direction": "long/short",
 "probability": %,
 "entry": number,
 "sl": number,
 "tp1": number,
 "tp2": number,
 "rr": number
}}

Symbol: {symbol}
Price: {price}
Snapshot: {json.dumps(snapshot)}
"""


# =============================
# GEMINI HELPERS
# =============================
def call_gemini(prompt):
    try:
        out = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return out.text
    except Exception as e:
        return None


def extract_json(txt):
    try:
        s = txt.index("{")
        e = txt.rindex("}")
        return json.loads(txt[s:e+1])
    except:
        return None


# =============================
# BINGX BALANCE
# =============================
def get_bingx_usdt_balance():
    try:
        data = bingx.swap_v2_get_balance()
        bals = data.get("data", {}).get("balance", [])
        for b in bals:
            if b["asset"] == "USDT":
                return float(b["availableBalance"])
    except:
        return None


# =============================
# AUTO TRADE EXECUTION
# =============================
def binance_to_bingx(symbol):
    return symbol.replace("USDT", "-USDT")


async def auto_trade(sig, context):
    symbol = sig["symbol"]
    if symbol not in SUPPORTED_BINGX:
        return

    if len(auto_positions) >= AUTO_MAX_POS:
        return

    balance = get_bingx_usdt_balance()
    if not balance or balance <= 0:
        await context.bot.send_message(OWNER_CHAT_ID, "⚠️ No balance for auto-trade")
        return

    entry = sig["entry"]
    qty = (balance * AUTO_LEVERAGE / AUTO_MAX_POS) / entry

    side = "LONG" if sig["direction"] == "long" else "SHORT"
    bingx_symbol = binance_to_bingx(symbol)

    try:
        order = bingx.open_market_order(
            bingx_symbol, side, f"{qty:.6f}",
            tp=str(sig["tp1"]),
            sl=str(sig["sl"])
        )

        auto_positions.add(symbol)

        await context.bot.send_message(
            OWNER_CHAT_ID,
            f"✅ Auto Trade Executed\n"
            f"{bingx_symbol}\nSide: {side}\nQty: {qty:.6f}\n"
            f"Entry:{entry}  SL:{sig['sl']}  TP:{sig['tp1']}"
        )
    except Exception as e:
        await context.bot.send_message(OWNER_CHAT_ID, f"❌ AutoTrade Error: {e}")


# =============================
# MANUAL ANALYSIS COMMAND
# =============================
async def analyze_manual(symbol, timeframe):
    tfs = [timeframe] if timeframe else DEFAULT_TFS
    snapshot, price = build_snapshot(symbol, tfs)

    if price is None:
        return f"❌ Could not fetch {symbol} candles"

    raw = call_gemini(prompt_manual(symbol, timeframe, snapshot, price))
    if not raw:
        return f"❌ Gemini request failed"

    data = extract_json(raw)
    if not data:
        return f"❌ JSON parsing error"

    prob = max(data.get("upside_probability",0), data.get("downside_probability",0))

    msg = f"📊 *{symbol} Analysis*\nPrice: `{price}`\nSummary: _{data.get('summary','')}_\n\n"
    msg += f"Upside: `{data['upside_probability']}%`\n"
    msg += f"Downside: `{data['downside_probability']}%`\n"
    msg += f"Flat: `{data['flat_probability']}%`\n"

    if prob >= MIN_PROB_MANUAL:
        msg += f"\nEntry:`{data['entry']}`\nSL:`{data['sl']}`\nTP1:`{data['tp1']}`\nTP2:`{data['tp2']}`"

    return msg


# =============================
# MANUAL HANDLE
# =============================
async def handle_pair(update:Update, context:ContextTypes.DEFAULT_TYPE):
    msg = update.message.text.strip().split()
    symbol = msg[0].replace("/", "").upper()
    timeframe = msg[1] if len(msg) > 1 else None

    await update.message.reply_text(f"⏳ Analysing {symbol}...")

    result = await analyze_manual(symbol, timeframe)
    await update.message.reply_markdown(result)


# =============================
# SCANNER ANALYSIS
# =============================
def analyze_scan(symbol):
    snapshot, price = build_snapshot(symbol, SCAN_TFS)
    if price is None:
        return None

    raw = call_gemini(prompt_scan(symbol, snapshot, price))
    if not raw:
        return None

    data = extract_json(raw)
    if not data:
        return None

    prob = int(data["probability"])
    rr = float(data["rr"])

    if prob < MIN_PROB_SCAN or rr < MIN_RR:
        return None

    return {
        "symbol": symbol,
        "direction": data["direction"],
        "probability": prob,
        "entry": float(data["entry"]),
        "sl": float(data["sl"]),
        "tp1": float(data["tp1"]),
        "tp2": float(data["tp2"]),
        "rr": rr
    }


# =============================
# SCANNER JOB
# =============================
async def scanner_job(context:ContextTypes.DEFAULT_TYPE):
    if not SCAN_ENABLED:
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
        if last and (now - last).total_seconds() < SIGNAL_COOLDOWN_SECONDS:
            continue

        last_signal_time[key] = now

        await context.bot.send_message(
            OWNER_CHAT_ID,
            f"🚨 *AI SIGNAL*\n"
            f"Symbol: `{sym}`\nDirection: `{sig['direction']}`\n"
            f"Prob:`{sig['probability']}%`\nRR:`{sig['rr']}`\n"
            f"Entry:`{sig['entry']}`\nSL:`{sig['sl']}`\nTP1:`{sig['tp1']}`",
            parse_mode="Markdown"
        )

        await auto_trade(sig, context)


# =============================
# START / STOP
# =============================
async def cmd_start(update, context):
    global SCAN_ENABLED
    SCAN_ENABLED = True
    await update.message.reply_text("✅ Auto Scanner ON (5 min scan / 10 min cooldown)")


async def cmd_stop(update, context):
    global SCAN_ENABLED
    SCAN_ENABLED = False
    await update.message.reply_text("⏹ Auto Scanner OFF")


# =============================
# MAIN
# =============================
async def main():
    load_supported_bingx_symbols()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(MessageHandler(filters.COMMAND, handle_pair))

    app.job_queue.run_repeating(scanner_job, interval=SCAN_INTERVAL, first=10)

    await app.run_polling()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

