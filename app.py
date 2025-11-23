import os
import json
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
# ENV & GLOBAL CONFIG
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

# Gemini models (no "models/" prefix with google-generativeai)
GEMINI_MODEL_MANUAL = os.getenv("GEMINI_MODEL_MANUAL", "gemini-1.5-flash")
GEMINI_MODEL_SCANNER = os.getenv("GEMINI_MODEL_SCANNER", GEMINI_MODEL_MANUAL)

# Scanner behaviour
SCAN_ENABLED = True
SCAN_INTERVAL_SECONDS = 300  # run every 5 minutes

MIN_VOLUME = 50_000_000       # Bybit 24h turnover filter
MAX_SCAN_SYMBOLS = 30

MIN_PROB_MANUAL = 80          # only give setup if >= 80%
MIN_PROB_SCAN = 80
MIN_RR = 1.9

DEFAULT_TIMEFRAMES = [
    "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"
]
SCAN_TIMEFRAMES = ["15m", "1h", "4h"]

# Auto-trade
AUTO_MAX_POSITIONS = 2
AUTO_LEVERAGE = 3.0
SIGNAL_COOLDOWN_SECONDS = 600   # 10 minutes per (symbol, direction)

last_signal_time = {}
auto_open_positions = set()
SUPPORTED_BINGX = set()

# ============================================================
# CLIENTS
# ============================================================

genai.configure(api_key=GEMINI_API_KEY)
gemini_manual = genai.GenerativeModel(GEMINI_MODEL_MANUAL)
gemini_scanner = genai.GenerativeModel(GEMINI_MODEL_SCANNER)

bingx = None
if BINGX_API_KEY and BINGX_API_SECRET:
    # demo=False -> real, set True if you want testnet behaviour
    bingx = BingxAPI(BINGX_API_KEY, BINGX_API_SECRET, demo=False, timestamp="local")

# ============================================================
# BYBIT MARKET DATA
# ============================================================

BYBIT_ENDPOINT = "https://api.bybit.com"

BYBIT_INTERVAL_MAP = {
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
    "1w": "W",
}


def fetch_bybit_candles(symbol: str, timeframe: str):
    """
    Fetch OHLCV from Bybit linear futures.
    Returns list of dicts with open_time, open, high, low, close, volume.
    """
    interval = BYBIT_INTERVAL_MAP.get(timeframe)
    if interval is None:
        return []

    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": 200,
    }

    for attempt in range(3):
        try:
            r = requests.get(f"{BYBIT_ENDPOINT}/v5/market/kline", params=params, timeout=10)
            data = r.json()
            if data.get("retCode") != 0:
                time.sleep(1.0 * (attempt + 1))
                continue

            klines = []
            # Bybit returns: [startTime, open, high, low, close, volume, turnover]
            for k in data.get("result", {}).get("list", []):
                klines.append(
                    {
                        "open_time": int(k[0]),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                    }
                )
            # reverse so oldest -> newest
            return list(reversed(klines))
        except Exception:
            time.sleep(1.0 * (attempt + 1))

    return []


def get_bybit_top_symbols():
    """Top USDT linear futures by 24h turnover, filtered by MIN_VOLUME."""
    params = {"category": "linear"}
    try:
        r = requests.get(f"{BYBIT_ENDPOINT}/v5/market/tickers", params=params, timeout=10)
        data = r.json()
        if data.get("retCode") != 0:
            return []

        items = data.get("result", {}).get("list", [])
        filtered = [
            it for it in items
            if it.get("symbol", "").endswith("USDT")
            and float(it.get("turnover24h", "0") or 0) >= MIN_VOLUME
        ]
        filtered.sort(key=lambda x: float(x.get("turnover24h", "0") or 0), reverse=True)
        return [it["symbol"] for it in filtered[:MAX_SCAN_SYMBOLS]]
    except Exception:
        return []


def get_latest_bybit_price(symbol: str):
    """Latest price from Bybit ticker."""
    params = {"category": "linear", "symbol": symbol}
    try:
        r = requests.get(f"{BYBIT_ENDPOINT}/v5/market/tickers", params=params, timeout=10)
        data = r.json()
        if data.get("retCode") != 0:
            return None
        lst = data.get("result", {}).get("list", [])
        if not lst:
            return None
        return float(lst[0].get("lastPrice"))
    except Exception:
        return None


# ============================================================
# BINGX HELPERS (SYMBOLS, BALANCE, ORDERS)
# ============================================================

def load_supported_bingx_symbols():
    """Fill SUPPORTED_BINGX with symbols like 'BTCUSDT' that exist on BingX."""
    global SUPPORTED_BINGX
    if not bingx:
        SUPPORTED_BINGX = set()
        return

    try:
        data = bingx.get_all_contracts()
        symbols = set()

        if isinstance(data, dict):
            contracts = (
                data.get("data", {}).get("contracts")
                or data.get("data")
                or data.get("contracts")
                or []
            )
        else:
            contracts = data or []

        for c in contracts:
            sym = None
            if isinstance(c, dict):
                sym = c.get("symbol") or c.get("pair")
            elif isinstance(c, str):
                sym = c

            if not sym:
                continue

            # normalise 'BTC-USDT' or 'BTCUSDT' -> 'BTCUSDT'
            if sym.endswith("-USDT"):
                symbols.add(sym.replace("-USDT", "USDT"))
            elif sym.endswith("USDT"):
                symbols.add(sym)

        SUPPORTED_BINGX = symbols
        print(f"[BINGX] Loaded {len(SUPPORTED_BINGX)} futures symbols.")
    except Exception as e:
        print("load_supported_bingx_symbols error:", e)
        SUPPORTED_BINGX = set()


def get_bingx_usdt_balance():
    """Return available margin in USDT from BingX perpetual account."""
    if not bingx:
        return None
    try:
        info = bingx.get_perpetual_balance()
        bal = (info.get("data") or {}).get("balance") or {}
        # Prefer availableMargin; fall back to balance
        avail = bal.get("availableMargin") or bal.get("balance")
        if avail is None:
            return None
        return float(avail)
    except Exception as e:
        print("get_bingx_usdt_balance error:", e)
        return None


def maybe_auto_trade(signal: dict, app):
    """
    Execute market order on BingX if:
      - BingX configured
      - symbol supported
      - open positions < AUTO_MAX_POSITIONS
    Uses 3x leverage and splits notional equally for up to 2 positions.
    """
    global auto_open_positions

    if not bingx:
        return

    symbol = signal["symbol"]
    direction = signal["direction"]

    if symbol not in SUPPORTED_BINGX:
        # Symbol not on BingX, just skip autotrade
        return

    if len(auto_open_positions) >= AUTO_MAX_POSITIONS:
        return

    balance = get_bingx_usdt_balance()
    if not balance or balance <= 0:
        if OWNER_CHAT_ID:
            app.create_task(
                app.bot.send_message(
                    OWNER_CHAT_ID,
                    "⚠️ Auto-trade skipped: BingX USDT balance unavailable or zero.",
                )
            )
        return

    entry = signal["entry"]
    if entry <= 0:
        return

    total_notional = balance * AUTO_LEVERAGE
    qty = (total_notional / AUTO_MAX_POSITIONS) / entry

    bingx_symbol = f"{symbol.replace('USDT', '')}-USDT"
    side = "LONG" if direction == "long" else "SHORT"

    try:
        bingx.open_market_order(
            bingx_symbol,
            side,
            round(qty, 4),
            tp=str(signal["tp1"]),
            sl=str(signal["sl"]),
        )
        auto_open_positions.add(symbol)

        if OWNER_CHAT_ID:
            text = (
                "✅ AutoTrade Executed\n"
                f"Symbol: {bingx_symbol}\n"
                f"Side: {side}\n"
                f"Qty: {qty:.4f}\n"
                f"Entry: {entry}\nSL: {signal['sl']}\nTP1: {signal['tp1']}"
            )
            app.create_task(app.bot.send_message(OWNER_CHAT_ID, text))
    except Exception as e:
        if OWNER_CHAT_ID:
            app.create_task(
                app.bot.send_message(OWNER_CHAT_ID, f"❌ AutoTrade Error: {e}")
            )


# ============================================================
# SNAPSHOT BUILDER
# ============================================================

def build_snapshot(symbol: str, timeframes):
    """
    Returns (snapshot_dict, current_price) using Bybit candles.
    Snapshot: { timeframe: [ {open, high, low, close, volume, open_time}, ... ] }
    """
    snapshot = {}
    current_price = None

    for tf in timeframes:
        candles = fetch_bybit_candles(symbol, tf)
        snapshot[tf] = candles
        if candles:
            current_price = candles[-1]["close"]

    if current_price is None:
        # Fallback: get current price from Bybit ticker
        current_price = get_latest_bybit_price(symbol)

    return snapshot, current_price


# ============================================================
# GEMINI PROMPTS & HELPERS
# ============================================================

def build_manual_prompt(symbol, timeframe, snapshot, price):
    """
    Prompt for /ETHUSDT [tf] manual analysis.
    We ask for tight JSON with direction, probabilities, RR, and reasoning.
    """
    tf_text = timeframe or "multi-timeframe (5m to 1W)"

    return f"""
You are a world-class crypto futures trader and risk manager.

Analyse the futures pair {symbol} using the multi-timeframe OHLCV snapshot below.
Key requirements:
- Timeframes available: {", ".join(snapshot.keys())}
- Current futures price: {price}
- Focus mainly on price action, key support/resistance, trendlines, VWAP behaviour,
  fixed range volume profile zones, and clear reversal candles (hammer, engulfing, doji).
- Consider overall trend, liquidity zones, and whether the move has a clean path to target.

1. First decide whether the next meaningful move on {tf_text} is:
   - "long" (upside move),
   - "short" (downside move), or
   - "flat" (no good trade).

2. Estimate probabilities (percent integers that sum to ~100):
   - upside_probability
   - downside_probability
   - flat_probability

3. Only if the dominant move (upside or downside) has probability >= {MIN_PROB_MANUAL},
   then propose a high-quality trade setup with:
   - entry (near current price but not random)
   - stop_loss at a logical key level (recent swing high/low or reversal candle high/low)
   - tp1 and tp2
   - risk-reward ratio "rr" based on entry vs best TP.
   - A very short one-sentence summary explaining WHY (e.g. bounce from VWAP + bullish engulfing from demand).

4. If the best move has probability < {MIN_PROB_MANUAL}, set entry, stop_loss, tp1, tp2 to null.

Return ONLY a single JSON object, no markdown, no explanation text, in this EXACT schema:

{{
  "symbol": "{symbol}",
  "timeframe": "{tf_text}",
  "direction": "long" | "short" | "flat",
  "summary": "very short explanation of the setup",
  "upside_probability": 0,
  "downside_probability": 0,
  "flat_probability": 0,
  "entry": null or 0.0,
  "stop_loss": null or 0.0,
  "tp1": null or 0.0,
  "tp2": null or 0.0,
  "rr": null or 0.0
}}

Snapshot (JSON) to analyse:
{json.dumps(snapshot)}
"""


def build_scan_prompt(symbol, snapshot, price):
    """
    Prompt for autoscan. We only care about high-conviction trades.
    """
    return f"""
You are an ultra-fast crypto futures scanner.

Analyse the pair {symbol} using the OHLCV snapshot below.
Goal: detect only very high conviction trades (probability >= {MIN_PROB_SCAN}% and RR >= {MIN_RR}).
Focus heavily on:
- VWAP reaction,
- fixed range volume profile zones between recent swing high/low,
- trend direction,
- strong reversal candles at key levels.

Rules:
- If there is NO clean, high-probability setup, respond with "direction": "flat" and set "probability" to 0 and all prices to null.
- Otherwise, choose "direction": "long" or "short".
- "probability" (integer 0-100) is the probability that price reaches TP1 before the stop_loss.
- Provide: entry, stop_loss, tp1, tp2 and "rr" >= {MIN_RR}.
- Also provide a very short "summary" (max 1 sentence) explaining the logic.

Return ONLY a JSON object with this schema:

{{
  "symbol": "{symbol}",
  "direction": "long" | "short" | "flat",
  "probability": 0,
  "rr": null or 0.0,
  "entry": null or 0.0,
  "stop_loss": null or 0.0,
  "tp1": null or 0.0,
  "tp2": null or 0.0,
  "summary": "very short explanation"
}}

Snapshot (JSON):
{json.dumps(snapshot)}

Current price: {price}
"""


def call_gemini(prompt: str, mode: str) -> str:
    """Call Gemini (manual or scanner model) and return raw text."""
    model = gemini_manual if mode == "manual" else gemini_scanner
    try:
        resp = model.generate_content(prompt)
        return resp.text or ""
    except Exception as e:
        print("call_gemini error:", e)
        return ""


def extract_json(text: str):
    """Extract first JSON object from model output."""
    if not text:
        return None
    try:
        start = text.index("{")
        end = text.rindex("}")
        raw = text[start: end + 1]
        return json.loads(raw)
    except Exception as e:
        print("extract_json error:", e, "text:", text[:200])
        return None


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def analyze_manual(symbol: str, timeframe: str | None):
    tfs = [timeframe] if timeframe else DEFAULT_TIMEFRAMES
    snapshot, price = build_snapshot(symbol, tfs)
    if price is None:
        return f"❌ Could not fetch candles for {symbol} from Bybit."

    prompt = build_manual_prompt(symbol, timeframe, snapshot, price)
    raw = call_gemini(prompt, mode="manual")
    data = extract_json(raw)
    if not data:
        return "❌ Gemini JSON error."

    direction = data.get("direction", "flat")
    up = int(data.get("upside_probability", 0))
    down = int(data.get("downside_probability", 0))
    flat = int(data.get("flat_probability", 0))
    summary = data.get("summary") or "-"

    lines = []
    if symbol not in SUPPORTED_BINGX:
        lines.append(
            f"ℹ️ {symbol} is not supported on BingX. Auto-trade will be skipped.\n"
        )

    lines += [
        f"📊 *{symbol} Analysis*",
        f"Price: `{price}`",
        f"Direction: *{direction}*",
        f"Upside: `{up}%`  Downside: `{down}%`  Flat: `{flat}%`",
        f"Reason: _{summary}_",
    ]

    # Only show trade levels if Gemini actually gave a setup
    entry = data.get("entry")
    sl = data.get("stop_loss")
    tp1 = data.get("tp1")
    tp2 = data.get("tp2")
    rr = data.get("rr")

    if entry and sl and tp1:
        lines += [
            "",
            f"Entry: `{entry}`",
            f"SL: `{sl}`",
            f"TP1: `{tp1}`",
        ]
        if tp2:
            lines.append(f"TP2: `{tp2}`")
        if rr:
            lines.append(f"RR: `{rr}`")

    return "\n".join(lines)


def analyze_for_scan(symbol: str):
    snapshot, price = build_snapshot(symbol, SCAN_TIMEFRAMES)
    if price is None:
        return None

    prompt = build_scan_prompt(symbol, snapshot, price)
    raw = call_gemini(prompt, mode="scanner")
    data = extract_json(raw)
    if not data:
        return None

    direction = data.get("direction", "flat")
    if direction == "flat":
        return None

    prob = int(data.get("probability", 0))
    rr = float(data.get("rr") or 0.0)
    if prob < MIN_PROB_SCAN or rr < MIN_RR:
        return None

    try:
        entry = float(data["entry"])
        sl = float(data["stop_loss"])
        tp1 = float(data["tp1"])
        tp2 = float(data.get("tp2") or tp1)
    except Exception:
        return None

    return {
        "symbol": symbol,
        "direction": direction,
        "probability": prob,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rr": rr,
        "summary": data.get("summary") or "",
    }


# ============================================================
# TELEGRAM HANDLERS
# ============================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SCAN_ENABLED
    SCAN_ENABLED = True
    await update.message.reply_text(
        "✅ Auto scanner ON.\n"
        "• Scans top Bybit USDT futures every 5 minutes.\n"
        "• Sends only signals with probability ≥ "
        f"{MIN_PROB_SCAN}% and RR ≥ {MIN_RR}.\n"
        "• Cooldown per pair/direction: 10 minutes."
    )


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SCAN_ENABLED
    SCAN_ENABLED = False
    await update.message.reply_text("⏹ Auto scanner OFF. Manual analysis still works.")


async def handle_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return

    text = update.message.text.strip()
    parts = text.split()

    if not parts:
        return

    symbol = parts[0].replace("/", "").upper()
    timeframe = parts[1] if len(parts) > 1 else None

    await update.message.reply_text(f"⏳ Analysing {symbol}...")

    try:
        result = await asyncio.to_thread(analyze_manual, symbol, timeframe)
    except Exception as e:
        result = f"❌ Error analysing {symbol}: {e}"

    await update.message.reply_markdown(result)


# ============================================================
# SCANNER LOOP
# ============================================================

async def run_scan_once(app):
    if not SCAN_ENABLED or not OWNER_CHAT_ID:
        return

    symbols = await asyncio.to_thread(get_bybit_top_symbols)
    if not symbols:
        return

    now = datetime.now(timezone.utc)

    for sym in symbols:
        try:
            sig = await asyncio.to_thread(analyze_for_scan, sym)
        except Exception as e:
            print("scan error for", sym, ":", e)
            continue

        if not sig:
            continue

        key = (sym, sig["direction"])
        last = last_signal_time.get(key)
        if last and (now - last).total_seconds() < SIGNAL_COOLDOWN_SECONDS:
            continue

        last_signal_time[key] = now

        warn = ""
        if sym not in SUPPORTED_BINGX:
            warn = (
                f"ℹ️ {sym} is not supported on BingX. Auto-trade will be skipped.\n\n"
            )

        msg = (
            warn +
            "🚨 *AI SIGNAL*\n"
            f"Symbol: `{sym}`\n"
            f"Direction: `{sig['direction']}`\n"
            f"Probability: `{sig['probability']}%`\n"
            f"RR: `{sig['rr']}`\n"
            f"Entry: `{sig['entry']}`\n"
            f"SL: `{sig['sl']}`\n"
            f"TP1: `{sig['tp1']}`\n"
            f"Reason: _{sig['summary']}_"
        )

        await app.bot.send_message(OWNER_CHAT_ID, msg, parse_mode="Markdown")

        # Auto-trade (sync; short API call)
        maybe_auto_trade(sig, app)


async def scanner_loop(app):
    """Background loop run inside the PTB event loop."""
    await asyncio.sleep(5)
    while True:
        try:
            await run_scan_once(app)
        except Exception as e:
            print("scanner_loop error:", e)
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)


async def post_init(app):
    # Load BingX symbols once at startup
    await asyncio.to_thread(load_supported_bingx_symbols)
    # Start background scanner task
    app.create_task(scanner_loop(app))


# ============================================================
# MAIN
# ============================================================

async def main():
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("stop", cmd_stop))
    application.add_handler(MessageHandler(filters.COMMAND, handle_pair))

    await application.run_polling()


if __name__ == "__main__":
    asyncio.run(main())
