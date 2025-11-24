# ============================================================
# TELEGRAM + GEMINI + BYBIT SCANNER + OPTIONAL BINGX AUTOTRADE
# - Manual analysis:  /btcusdt , /ethusdt , etc.
# - Auto-scan: scans top Bybit USDT futures every N minutes.
# - Uses python-telegram-bot v13 (Updater + JobQueue, no webhooks).
# ============================================================

import os
import json
import re
import time

import requests
import google.generativeai as genai

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

# Use a stable model name for google-generativeai.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

# Scanner / trade settings
SCAN_INTERVAL_SECONDS = 300          # background scan interval (5 min)
SIGNAL_COOLDOWN_SECONDS = 600        # 10 min cooldown per (symbol, direction)
MIN_VOLUME = 50_000_000              # Bybit 24h turnover filter
MIN_PROB_SCAN = 80                   # autoscan probability threshold
MIN_RR = 1.9                         # minimum RR

AUTO_LEVERAGE = 3
AUTO_MAX_POSITIONS = 2

SCAN_ENABLED = True
SUPPORTED_BINGX = set()
auto_open_positions = set()
last_signal_time = {}  # (symbol, direction) -> last timestamp

# ============================================================
# CLIENTS
# ============================================================

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

bingx = None
if BINGX_API_KEY and BINGX_API_SECRET:
    try:
        bingx = BingxAPI(BINGX_API_KEY, BINGX_API_SECRET)
    except Exception as e:
        print("BingxAPI init error:", e)
        bingx = None

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
    """Returns list of top USDT linear futures symbols by 24h turnover."""
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
    """Returns OHLCV candles for Bybit linear futures."""
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
    """Latest price from Bybit ticker."""
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
# BINGX HELPERS
# ============================================================

def load_supported_bingx_symbols():
    """Fill SUPPORTED_BINGX with symbols like 'BTCUSDT' that exist on BingX."""
    global SUPPORTED_BINGX

    if not bingx:
        SUPPORTED_BINGX = set()
        return

    try:
        data = bingx.get_all_contracts()
    except Exception as e:
        print("load_supported_bingx_symbols error:", e)
        SUPPORTED_BINGX = set()
        return

    if isinstance(data, dict):
        contracts = (
            data.get("data", {}).get("contracts")
            or data.get("data")
            or data.get("contracts")
            or []
        )
    else:
        contracts = data or []

    symbols = set()
    for c in contracts:
        if isinstance(c, dict):
            sym = c.get("symbol") or c.get("pair")
        else:
            sym = str(c)
        if not sym:
            continue
        if sym.endswith("-USDT"):
            symbols.add(sym.replace("-USDT", "USDT"))
        elif sym.endswith("USDT"):
            symbols.add(sym)

    SUPPORTED_BINGX = symbols
    print(f"[BINGX] Loaded {len(SUPPORTED_BINGX)} symbols")


def get_bingx_usdt_balance():
    """Return available margin in USDT from BingX perpetual account."""
    if not bingx:
        return None
    try:
        info = bingx.get_perpetual_balance()
        bal = (info.get("data") or {}).get("balance") or {}
        avail = bal.get("availableMargin") or bal.get("balance")
        if avail is None:
            return None
        return float(avail)
    except Exception as e:
        print("get_bingx_usdt_balance error:", e)
        return None


# ============================================================
# GEMINI HELPERS — BULLETPROOF JSON + TEXT FALLBACK
# ============================================================

def force_json(text: str):
    """
    Extract first valid JSON object from text.
    Very defensive: direct parse, bracket slice, regex.
    """
    if not text:
        return {}

    # Try direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", text)
    cleaned = cleaned.replace("```", "")

    # Try cleaned
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Bracket slice
    try:
        start = cleaned.index("{")
        end = cleaned.rindex("}")
        return json.loads(cleaned[start:end + 1])
    except Exception:
        pass

    # Regex
    try:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass

    return {}


def ask_gemini_json(prompt: str):
    """Call Gemini and return parsed JSON (or {})."""
    try:
        resp = gemini_model.generate_content(prompt)
        txt = (resp.text or "").strip()
        return force_json(txt)
    except Exception as e:
        print("Gemini JSON error:", e)
        return {}


def ask_gemini_text(prompt: str) -> str:
    """Call Gemini and return plain text (no JSON)."""
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

Focus on trend, key levels, VWAP, volume profile and reversal candles.
Decide ONLY if there is a very clean setup.

Return ONLY valid JSON:

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
   propose entry, stop, tp1, tp2 based on key levels.
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

def analyze_manual(symbol: str) -> str:
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
    data = ask_gemini_json(prompt)

    if not data:
        # Fallback plain-text explanation
        fallback_prompt = f"""
You are a top crypto trader.

Give a short 3–5 line trading view for {symbol} at price {price}.
Talk about trend, key support/resistance, and whether longs/shorts/flat
are better. Do NOT return JSON, just text.
"""
        text = ask_gemini_text(fallback_prompt)
        if not text:
            return (
                f"❌ Gemini could not analyse {symbol} right now.\n"
                f"Model: {GEMINI_MODEL}\n"
                f"Please try again in a few seconds."
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
# AUTOTRADE
# ============================================================

def maybe_autotrade(signal, bot):
    if not bingx:
        return

    sym = signal["symbol"]
    direction = signal["direction"]

    if sym not in SUPPORTED_BINGX:
        print(f"{sym} not supported on BingX — skip autotrade.")
        return

    if len(auto_open_positions) >= AUTO_MAX_POSITIONS:
        print("Max positions reached, skip autotrade.")
        return

    balance = get_bingx_usdt_balance()
    if not balance or balance <= 0:
        if OWNER_CHAT_ID:
            bot.send_message(
                OWNER_CHAT_ID,
                "⚠️ AutoTrade skipped: BingX USDT balance unavailable or zero.",
            )
        return

    entry = signal["entry"]
    if not entry or entry <= 0:
        return

    notional = balance * AUTO_LEVERAGE
    qty = round((notional / AUTO_MAX_POSITIONS) / entry, 4)

    bingx_symbol = f"{sym.replace('USDT', '')}-USDT"
    side = "LONG" if direction == "long" else "SHORT"

    try:
        bingx.open_market_order(
            bingx_symbol,
            side,
            qty,
            tp=str(signal["tp1"]),
            sl=str(signal["stop"]),
        )
        auto_open_positions.add(sym)

        if OWNER_CHAT_ID:
            bot.send_message(
                OWNER_CHAT_ID,
                f"✅ AutoTrade Executed\n"
                f"Symbol: {bingx_symbol}\n"
                f"Side: {side}\n"
                f"Qty: {qty}\n"
                f"Entry: {entry}\n"
                f"SL: {signal['stop']}\n"
                f"TP1: {signal['tp1']}",
            )
    except Exception as e:
        print("Autotrade error:", e)
        if OWNER_CHAT_ID:
            bot.send_message(OWNER_CHAT_ID, f"❌ AutoTrade error: {e}")


# ============================================================
# AUTOSCAN LOGIC
# ============================================================

def analyze_signal(symbol: str):
    price = get_price(symbol)
    if price is None:
        return None

    candles = {
        "15m": get_candles(symbol, "15m"),
        "1h": get_candles(symbol, "1h"),
        "4h": get_candles(symbol, "4h"),
    }

    prompt = build_scan_prompt(symbol, candles, price)
    data = ask_gemini_json(prompt)
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


def scan_job(context: CallbackContext):
    """JobQueue callback: runs periodically in background."""
    global last_signal_time

    if not SCAN_ENABLED or not OWNER_CHAT_ID:
        return

    symbols = get_bybit_symbols()
    if not symbols:
        return

    now = time.time()

    for sym in symbols:
        try:
            sig = analyze_signal(sym)
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

        warn = ""
        if sym not in SUPPORTED_BINGX:
            warn = f"ℹ️ {sym} not supported on BingX → autotrade OFF.\n\n"

        msg = (
            warn
            + f"🚨 *AI SIGNAL*\n"
            f"Symbol: `{sym}`\n"
            f"Direction: `{sig['direction']}`\n"
            f"Probability: `{sig['probability']}%`\n"
            f"RR: `{sig['rr']}`\n"
            f"Entry: `{sig['entry']}`\n"
            f"SL: `{sig['stop']}`\n"
            f"TP1: `{sig['tp1']}`\n"
            f"Reason: _{sig['summary']}_"
        )

        context.bot.send_message(OWNER_CHAT_ID, msg, parse_mode="Markdown")

        # auto-trade
        maybe_autotrade(sig, context.bot)


# ============================================================
# TELEGRAM HANDLERS
# ============================================================

def cmd_start(update: Update, context: CallbackContext):
    global SCAN_ENABLED
    SCAN_ENABLED = True
    update.message.reply_text(
        "✅ Auto-scanner ON.\n"
        f"Scans top Bybit USDT futures every {SCAN_INTERVAL_SECONDS // 60} minutes.\n"
        f"Signals only if probability ≥ {MIN_PROB_SCAN}% and RR ≥ {MIN_RR}."
    )


def cmd_stop(update: Update, context: CallbackContext):
    global SCAN_ENABLED
    SCAN_ENABLED = False
    update.message.reply_text(
        "⏹ Auto-scanner OFF. Manual analysis still works."
    )


def handle_pair(update: Update, context: CallbackContext):
    """Handles commands like /btcusdt or /ethusdt."""
    if not update.message:
        return

    text = update.message.text.strip()

    # Remove leading "/" and any arguments after a space
    if text.startswith("/"):
        symbol = text[1:].split()[0].upper()
    else:
        symbol = text.replace("/", "").split()[0].upper()

    if not symbol.endswith("USDT"):
        update.message.reply_text(
            "Send coin like: `/btcusdt` or `/ethusdt`", parse_mode="Markdown"
        )
        return

    update.message.reply_text(f"⏳ Analysing {symbol}...")

    try:
        result = analyze_manual(symbol)
    except Exception as e:
        print("handle_pair error:", e)
        result = f"❌ Error analysing {symbol}: {e}"

    update.message.reply_markdown(result)


# ============================================================
# MAIN
# ============================================================

def main():
    # Updater style (v13) – stable and simple
    updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)

    # Load BingX symbols once at startup
    load_supported_bingx_symbols()

    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", cmd_start))
    dp.add_handler(CommandHandler("stop", cmd_stop))
    dp.add_handler(
        MessageHandler(
            Filters.command & ~Filters.regex(r"^/(start|stop)$"),
            handle_pair,
        )
    )

    # Background scanner via JobQueue
    job_queue = updater.job_queue
    job_queue.run_repeating(scan_job, interval=SCAN_INTERVAL_SECONDS, first=10)

    print("Bot running with polling (v13 Updater)...")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
