# ============================================================
# FULL APP.PY — BACKGROUND SCANNER + MANUAL ANALYSIS
# Bybit data + Gemini 2.5 + BingX autotrade
# NO JobQueue, works on Render with PTB 20+
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

# Use latest Gemini 2.5 Flash experimental by default
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-exp")

# scanner / trade settings
SCAN_INTERVAL_SECONDS = 300          # background scan interval
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
    # Try a few common constructor styles to be tolerant to versions
    try:
        bingx = BingxAPI(BINGX_API_KEY, BINGX_API_SECRET)
    except TypeError:
        try:
            bingx = BingxAPI(BINGX_API_KEY, BINGX_API_SECRET, timestamp="local")
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
# BINGX HELPERS
# ============================================================

def load_supported_bingx_symbols():
    """
    Fill SUPPORTED_BINGX with symbols like 'BTCUSDT' that exist on BingX.
    """
    global SUPPORTED_BINGX

    if not bingx:
        SUPPORTED_BINGX = set()
        return

    try:
        data = bingx.get_all_contracts()
    except AttributeError:
        # fallback to older method names if needed
        try:
            data = bingx.swap_v2_get_contracts()
        except Exception as e:
            print("load_supported_bingx_symbols error:", e)
            SUPPORTED_BINGX = set()
            return
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
    """
    Return available margin in USDT from BingX perpetual account.
    """
    if not bingx:
        return None
    try:
        try:
            info = bingx.get_perpetual_balance()
        except AttributeError:
            info = bingx.swap_v2_get_balance()

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
    Very defensive: tries direct parse, bracket slicing, regex.
    If everything fails, returns {} instead of raising.
    """
    if not text:
        return {}

    # try direct JSON first
    try:
        return json.loads(text)
    except Exception:
        pass

    # strip markdown fences like ```json ... ``` / ```
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
    Uses response_mime_type='application/json' to force JSON output.
    """
    try:
        resp = gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.25,
                "top_p": 0.9,
                "max_output_tokens": 2048,
                "response_mime_type": "application/json",
            },
        )
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
        resp = gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.4,
                "top_p": 0.9,
                "max_output_tokens": 512,
                "response_mime_type": "text/plain",
            },
        )
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

Focus on:
- Trend direction (uptrend, downtrend, ranging)
- Key support & resistance zones
- VWAP behaviour
- Fixed range volume profile zones (high volume nodes, low volume nodes)
- Reversal candles (hammer, shooting star, engulfing, doji) at key levels.

We only want VERY CLEAN setups.

Rules:
- If there is no clean, high-probability setup, respond with:
  "direction": "flat",
  "probability": 0,
  and set "entry", "stop", "tp1" to 0.
- If there IS a clean setup:
  * direction must be "long" or "short"
  * probability must be between 80 and 100
  * rr must be at least {MIN_RR} (RR = distance to TP / distance to stop)

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
You are a world-class crypto futures trader.

Symbol: {symbol}
Current price: {price}
Snapshot (multi-timeframe OHLCV JSON): {json.dumps(snapshot)}

Steps:
1. Evaluate the probabilities (integer 0–100) for:
   - upside (price moves up and breaks recent swing highs)
   - downside (price moves down and breaks recent swing lows)
   - flat (choppy, no edge, avoid trading)
   These should roughly sum to 100.

2. Choose "direction":
   - "long" if upside is clearly best,
   - "short" if downside is clearly best,
   - "flat" if there is no edge or price is choppy.

3. If direction is "long" or "short" AND its probability >= 80,
   propose:
   - entry: near current price but not random,
   - stop: at a logical key level (swing high/low, reversal candle, clear S/R),
   - tp1 and tp2: realistic targets in the direction of the trade.

4. If there is NO good trade (best probability < 80), set entry, stop, tp1, tp2 to null.

Return ONLY JSON in this exact schema:

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
            return (
                f"❌ AI could not generate a structured analysis for {symbol} right now.\n"
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
# AUTOTRADE (simple always-execute version)
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
        # try modern open_market_order
        try:
            bingx.open_market_order(
                bingx_symbol,
                side,
                qty,
                tp=str(signal["tp1"]),
                sl=str(signal["stop"]),
            )
        except AttributeError:
            # fallback to older swap_v2_place_order style if needed
            bingx.swap_v2_place_order(
                symbol=bingx_symbol,
                side="BUY" if side == "LONG" else "SELL",
                positionSide="LONG" if side == "LONG" else "SHORT",
                type="MARKET",
                quantity=str(qty),
                takeProfit=str(signal["tp1"]),
                stopLoss=str(signal["stop"]),
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

        await app.bot.send_message(OWNER_CHAT_ID, msg, parse_mode="Markdown")

        # auto-trade
        maybe_autotrade(sig, app.bot)


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
        result = f"❌ Error analysing {symbol}: {e}"

    await update.message.reply_markdown(result)


# ============================================================
# POST_INIT + MAIN
# ============================================================

async def post_init(app):
    # load BingX symbols once (in background thread)
    await asyncio.to_thread(load_supported_bingx_symbols)
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
