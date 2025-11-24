# ============================================================
# TELEGRAM BOT + GEMINI AI + BYBIT SCANNER (NO AUTOTRADE)
# ------------------------------------------------------------
# - Deploy as a Web Service on Render
# - Uses polling (no webhooks)
# - Manual analysis: /btcusdt, /ethusdt, etc.
# - Auto-scan: scans top Bybit USDT futures every N minutes
#   and sends best signals to OWNER_CHAT_ID
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
    filters,
)

# ============================================================
# ENVIRONMENT
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "0") or "0")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")

# Use a stable, supported model name.
# Keep this as gemini-1.5-pro unless you are 100% sure.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

# Scanner / analysis settings
SCAN_INTERVAL_SECONDS = 300          # background scan interval (5 min)
SIGNAL_COOLDOWN_SECONDS = 600        # 10 min cooldown per (symbol, direction)
MIN_VOLUME = 50_000_000              # Bybit 24h turnover filter
MIN_PROB_SCAN = 80                   # autoscan probability threshold
MIN_RR = 1.9                         # minimum RR

SCAN_ENABLED = True
last_signal_time: dict[tuple[str, str], float] = {}  # (symbol, direction) -> last timestamp

# ============================================================
# CLIENTS
# ============================================================

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

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
    """
    Returns list of top USDT linear futures symbols by 24h turnover.
    """
    try:
        r = requests.get(
            f"{BYBIT_ENDPOINT}/v5/market/tickers",
            params={"category": "linear"},
            timeout=10,
        )
        data = r.json()
        items = data.get("result", {}).get("list", []) or []
        filtered = [
            it
            for it in items
            if it.get("symbol", "").endswith("USDT")
            and float(it.get("turnover24h") or 0) >= MIN_VOLUME
        ]
        filtered.sort(key=lambda x: float(x.get("turnover24h") or 0), reverse=True)
        return [it["symbol"] for it in filtered[:30]]
    except Exception as e:
        print("get_bybit_symbols error:", e)
        return []


def get_candles(symbol: str, tf: str):
    """
    Returns OHLCV candles for Bybit linear futures.
    """
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


def get_price(symbol: str):
    """
    Latest price from Bybit ticker.
    """
    try:
        r = requests.get(
            f"{BYBIT_ENDPOINT}/v5/market/tickers",
            params={"category": "linear", "symbol": symbol},
            timeout=10,
        )
        data = r.json()
        items = data.get("result", {}).get("list", []) or []
        if not items:
            return None
        return float(items[0].get("lastPrice"))
    except Exception as e:
        print("get_price error:", e)
        return None


# ============================================================
# GEMINI HELPERS — BULLETPROOF JSON + TEXT FALLBACK
# ============================================================

def force_json(text: str):
    """
    Extract first valid JSON object from text.
    Very defensive: tries direct parse, cleaned parse, slicing, regex.
    If everything fails, returns {} instead of raising.
    """
    if not text:
        return {}

    # try direct JSON first
    try:
        return json.loads(text)
    except Exception:
        pass

    # strip markdown fences like ```json ... ```
    cleaned = re.sub(r"```(?:json)?", "", text)
    cleaned = cleaned.replace("```", "")

    # try again on cleaned
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # bracket slice
    try:
        start = cleaned.index("{")
        end = cleaned.rindex("}")
        return json.loads(cleaned[start:end + 1])
    except Exception:
        pass

    # regex fallback
    try:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass

    return {}


def ask_gemini_json(prompt: str):
    """
    Call Gemini and return parsed JSON (or {}).
    If the model name is invalid (404) or any error occurs,
    we log and return {} so caller can fall back.
    """
    try:
        resp = gemini_model.generate_content(prompt)
        txt = (resp.text or "").strip()
        return force_json(txt)
    except Exception as e:
        print("Gemini JSON error:", e)
        return {}


def ask_gemini_text(prompt: str) -> str:
    """
    Call Gemini and return plain text summary (no JSON).
    Used as fallback when JSON is unusable.
    """
    try:
        resp = gemini_model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        print("Gemini text error:", e)
        return ""


# ============================================================
# PROMPTS
# ============================================================

def build_scan_prompt(symbol, candles, price):
    return f"""
You are an expert crypto futures analyst.

Analyse:
- Symbol: {symbol}
- Current price: {price}
- Candles JSON: {json.dumps(candles)}

Focus on trend, key levels, VWAP behaviour, volume profile and reversal candles.
Decide ONLY if there is a very clean setup.

Return ONLY valid JSON in this exact schema:

{{
 "symbol": "{symbol}",
 "direction": "long" | "short" | "flat",
 "probability": 0,
 "rr": 0.0,
 "entry": 0.0,
 "stop": 0.0,
 "tp1": 0.0,
 "summary": "very short reason"
}}
"""


def build_manual_prompt(symbol, snapshot, price):
    return f"""
You are a world-class crypto trader.

Symbol: {symbol}
Current price: {price}
Snapshot: {json.dumps(snapshot)}

1. Evaluate upside, downside and flat probabilities (0-100 each).
2. Choose "direction": "long", "short" or "flat".
3. If direction is long/short AND its probability >= 80,
   propose entry, stop, tp1, tp2 based on key levels
   (recent swing high/low, reversal candle, strong support/resistance).
4. If no good trade, set entry/stop/tp1/tp2 to null.

Return ONLY JSON:

{{
 "symbol": "{symbol}",
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
"""


# ============================================================
# MANUAL ANALYSIS ( /btcusdt etc. )
# ============================================================

async def analyze_manual(symbol: str) -> str:
    symbol = symbol.upper()
    price = get_price(symbol)
    if price is None:
        return f"❌ Could not fetch price for {symbol}."

    snapshot = {
        "15m": get_candles(symbol, "15m"),
        "1h": get_candles(symbol, "1h"),
        "4h": get_candles(symbol, "4h"),
    }

    prompt = build_manual_prompt(symbol, snapshot, price)
    data = await asyncio.to_thread(ask_gemini_json, prompt)

    # If Gemini didn't give usable JSON, fall back to plain text explanation
    if not data:
        fallback_prompt = f"""
You are a top crypto trader.

Give a short 3–5 line trading view for {symbol} at price {price}.
Use the idea of trend, key support/resistance, and whether it is better
to look for longs, shorts or stay flat. Do NOT return JSON, just text.
"""
        text = await asyncio.to_thread(ask_gemini_text, fallback_prompt)
        if not text:
            # Last resort: at least tell user there was an AI issue
            return (
                f"❌ Gemini could not analyse {symbol} right now.\n"
                f"Model: {GEMINI_MODEL}\n"
                f"Please try again later."
            )

        return (
            f"📊 *{symbol} Analysis (fallback)*\n"
            f"Price: `{price}`\n\n"
            f"{text}"
        )

    direction = data.get("direction", "flat")
    summary = data.get("summary", "-")
    up = int(data.get("upside", 0))
    down = int(data.get("downside", 0))
    flat = int(data.get("flat", 0))

    lines = [
        f"📊 *{symbol} Analysis*",
        f"Price: `{price}`",
        f"Direction: *{direction}*",
        f"Upside: `{up}%`  Downside: `{down}%`  Flat: `{flat}%`",
        f"Reason: _{summary}_",
    ]

    entry = data.get("entry")
    stop = data.get("stop")
    tp1 = data.get("tp1")
    tp2 = data.get("tp2")

    if entry and stop and tp1:
        lines += [
            "",
            f"Entry: `{entry}`",
            f"SL: `{stop}`",
            f"TP1: `{tp1}`",
        ]
        if tp2:
            lines.append(f"TP2: `{tp2}`")

    return "\n".join(lines)


# ============================================================
# AUTOSCAN LOGIC (NO AUTOTRADE)
# ============================================================

async def analyze_signal(symbol: str):
    price = get_price(symbol)
    if price is None:
        return None

    candles = {
        "15m": get_candles(symbol, "15m"),
        "1h": get_candles(symbol, "1h"),
        "4h": get_candles(symbol, "4h"),
    }

    prompt = build_scan_prompt(symbol, candles, price)
    data = await asyncio.to_thread(ask_gemini_json, prompt)
    if not data:
        return None

    if data.get("direction") == "flat":
        return None
    if int(data.get("probability", 0)) < MIN_PROB_SCAN:
        return None
    if float(data.get("rr", 0.0)) < MIN_RR:
        return None

    try:
        entry = float(data["entry"])
        stop = float(data["stop"])
        tp1 = float(data["tp1"])
    except Exception:
        return None

    return {
        "symbol": symbol,
        "direction": data["direction"],
        "probability": int(data["probability"]),
        "entry": entry,
        "stop": stop,
        "tp1": tp1,
        "rr": float(data["rr"]),
        "summary": data.get("summary", ""),
    }


async def scan_once(app):
    if not SCAN_ENABLED or not OWNER_CHAT_ID:
        return

    symbols = await asyncio.to_thread(get_bybit_symbols)
    if not symbols:
        return

    now = time.time()

    for sym in symbols:
        try:
            sig = await analyze_signal(sym)
        except Exception as e:
            print("scan error for", sym, ":", e)
            continue

        if not sig:
            continue

        key = (sym, sig["direction"])
        last = last_signal_time.get(key)
        if last and now - last < SIGNAL_COOLDOWN_SECONDS:
            continue

        last_signal_time[key] = now

        msg = (
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

        await app.bot.send_message(OWNER_CHAT_ID, msg, parse_mode="Markdown")


async def scanner_loop(app):
    """
    Background loop started from post_init.
    No JobQueue used.
    """
    await asyncio.sleep(5)
    while True:
        try:
            await scan_once(app)
        except Exception as e:
            print("scanner_loop error:", e)
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)


# ============================================================
# TELEGRAM HANDLERS
# ============================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SCAN_ENABLED
    SCAN_ENABLED = True
    await update.message.reply_text(
        "✅ Auto-scanner ON.\n"
        f"Scans top Bybit USDT futures every {SCAN_INTERVAL_SECONDS // 60} minutes.\n"
        f"Signals only if probability ≥ {MIN_PROB_SCAN}% and RR ≥ {MIN_RR}."
    )


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SCAN_ENABLED
    SCAN_ENABLED = False
    await update.message.reply_text(
        "⏹ Auto-scanner OFF. Manual analysis still works."
    )


async def handle_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles commands like /btcusdt or /ethusdt.
    """
    if not update.message:
        return

    text = update.message.text.strip()

    # Remove leading "/" and any arguments after a space
    if text.startswith("/"):
        symbol = text[1:].split()[0].upper()
    else:
        symbol = text.replace("/", "").split()[0].upper()

    if not symbol.endswith("USDT"):
        await update.message.reply_text(
            "Send coin like: `/btcusdt` or `/ethusdt`", parse_mode="Markdown"
        )
        return

    await update.message.reply_text(f"⏳ Analysing {symbol}...")

    try:
        result = await analyze_manual(symbol)
    except Exception as e:
        print("handle_pair error:", e)
        result = f"❌ Error analysing {symbol}: {e}"

    await update.message.reply_markdown(result)


# ============================================================
# POST_INIT + MAIN
# ============================================================

async def post_init(app):
    # start scanner loop
    app.create_task(scanner_loop(app))


def main():
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("stop", cmd_stop))

    # Any other command → treated as coin, e.g. /btcusdt
    application.add_handler(
        MessageHandler(
            filters.COMMAND & ~filters.Regex(r"^/(start|stop)$"),
            handle_pair,
        )
    )

    print("Bot running...")
    application.run_polling()


if __name__ == "__main__":
    main()

