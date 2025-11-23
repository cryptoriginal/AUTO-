# FULL APP.PY — STABLE SCANNER + MANUAL ANALYSIS + BINGX AUTOTRADE
# JSON-SAFE VERSION (Option B timeframes: 15m, 1h, 4h, 1d)
# ---------------------------------------------------------------

import os
import json
import time
import asyncio
import re

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

# Scanner & trading settings
SCAN_INTERVAL_SECONDS = 300          # 5 minutes
SIGNAL_COOLDOWN_SECONDS = 600        # 10 minutes per (symbol, direction)
MIN_PROB_SCAN = 80
MIN_RR = 1.9
MIN_VOLUME = 50_000_000

AUTO_LEVERAGE = 3
AUTO_MAX_POSITIONS = 2

SCAN_ENABLED = True
SUPPORTED_BINGX = set()
auto_open_positions = set()
last_signal_time = {}
pending_manual_signals = {}  # chat_id -> signal dict (for future /yes flow if needed)

# ============================================================
# CLIENTS
# ============================================================

genai.configure(api_key=GEMINI_API_KEY)

# IMPORTANT: tell Gemini to answer in JSON directly
gemini_model = genai.GenerativeModel(
    GEMINI_MODEL,
    generation_config={
        "response_mime_type": "application/json",
    },
)

bingx = None
if BINGX_API_KEY and BINGX_API_SECRET:
    # Simple init – matches py-bingx
    bingx = BingxAPI(BINGX_API_KEY, BINGX_API_SECRET)

# ============================================================
# BYBIT MARKET DATA
# ============================================================

BYBIT_ENDPOINT = "https://api.bybit.com"

INTERVAL_MAP = {
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

# Option B: medium set of TFs for manual analysis (more stable)
MANUAL_TFS_DEFAULT = ["15m", "1h", "4h", "1d"]


def get_bybit_symbols():
    """Top linear USDT futures symbols by 24h turnover."""
    try:
        r = requests.get(
            f"{BYBIT_ENDPOINT}/v5/market/tickers",
            params={"category": "linear"},
            timeout=10,
        )
        data = r.json()
        lst = data.get("result", {}).get("list", []) or []
        lst = [x for x in lst if float(x.get("turnover24h", "0")) >= MIN_VOLUME]
        lst.sort(key=lambda x: float(x.get("turnover24h", "0")), reverse=True)
        return [x["symbol"] for x in lst[:20]]
    except Exception as e:
        print("get_bybit_symbols error:", e)
        return []


def get_candles(symbol: str, tf: str):
    """Fetch OHLCV candles from Bybit linear futures."""
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
    """Latest price from Bybit linear ticker."""
    try:
        r = requests.get(
            f"{BYBIT_ENDPOINT}/v5/market/tickers",
            params={"category": "linear", "symbol": symbol},
            timeout=10,
        )
        d = r.json()
        x = d.get("result", {}).get("list", []) or []
        if not x:
            return None
        return float(x[0].get("lastPrice"))
    except Exception as e:
        print("get_price error:", e)
        return None


# ============================================================
# GEMINI HELPERS (JSON-SAFE)
# ============================================================

def run_gemini(prompt: str) -> str:
    """Call Gemini and return raw text (should already be JSON)."""
    try:
        resp = gemini_model.generate_content(prompt)
        # because response_mime_type="application/json", this should be pure JSON
        return resp.text or ""
    except Exception as e:
        print("run_gemini error:", e)
        return ""


def clean_json_text(txt: str) -> str:
    """
    Remove code fences and junk around JSON.
    Works for outputs like:
    ```json
    { ... }
    ```
    or with extra text before/after.
    """
    if not txt:
        return ""

    # remove code fences ```json ... ``` or ``` ...
    txt = re.sub(r"```(?:json)?", "", txt, flags=re.IGNORECASE)
    txt = txt.replace("```", "")

    # trim whitespace
    txt = txt.strip()

    # if it's already starting with '{' and ending with '}', keep it
    if txt.startswith("{") and txt.endswith("}"):
        return txt

    # otherwise, try to grab first {...} block
    m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
    if m:
        return m.group(0).strip()

    return txt


def extract_json(txt: str):
    """
    Robust JSON extractor: cleans text and then tries json.loads.
    If it still fails, returns None instead of throwing.
    """
    if not txt:
        return None

    cleaned = clean_json_text(txt)
    try:
        return json.loads(cleaned)
    except Exception as e:
        print("extract_json error:", e, "cleaned:", cleaned[:200])
        return None


# ============================================================
# SCANNER PROMPT
# ============================================================

def scan_prompt(symbol, candles, price):
    return f"""
You are an expert crypto futures analyst.

Analyse the following Bybit linear futures data and propose only
VERY high probability trades.

Symbol: {symbol}
Current price: {price}
Candles JSON (15m,1h,4h): {json.dumps(candles)}

Rules:
- If you do NOT see a clean setup, respond with "direction": "flat", probability 0.
- Otherwise choose "long" or "short".
- "probability" = chance TP1 hits before SL (0-100).
- Only output trades with probability >= {MIN_PROB_SCAN} and RR >= {MIN_RR}.
- Use logical entry/stop/tp based on key levels, not random.
- Keep "summary" extremely short (one sentence max).

Return ONLY a single JSON object exactly like:

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
"""


# ============================================================
# MANUAL ANALYSIS PROMPT
# ============================================================

def manual_prompt(symbol, timeframe, snapshot, price):
    tf_text = timeframe or "multi-timeframe (15m, 1h, 4h, 1d)"

    return f"""
You are a world-class crypto futures trader and risk manager.

Analyse the futures pair {symbol} using this OHLCV snapshot:
Timeframes: {", ".join(snapshot.keys())}
Current futures price: {price}

Focus on:
- price action,
- key support/resistance and trend structure,
- VWAP behaviour and fixed range volume profile zones between recent swing high/low,
- clear reversal candles (hammer, engulfing, doji) at important levels.

1. Decide the next clean move on {tf_text}:
   - "long" (upside),
   - "short" (downside),
   - "flat" (no good trade).

2. Estimate probabilities (sum ~100):
   - upside_probability
   - downside_probability
   - flat_probability

3. Only if the dominant move (up/down) has probability >= {MIN_PROB_SCAN},
   propose a trade with:
   - entry (near current price),
   - stop at logical key level (swing high/low or S/R),
   - tp1 and tp2,
   - rr (risk reward vs best TP),
   - ultra-short summary explaining WHY.

If no clean setup, use direction="flat" and set entry/stop/tp1/tp2 to null.

Return ONLY ONE JSON object in this schema:

{{
  "symbol": "{symbol}",
  "timeframe": "{tf_text}",
  "direction": "long" | "short" | "flat",
  "summary": "very short explanation of the setup",
  "upside_probability": 0,
  "downside_probability": 0,
  "flat_probability": 0,
  "entry": null or 0.0,
  "stop": null or 0.0,
  "tp1": null or 0.0,
  "tp2": null or 0.0,
  "rr": null or 0.0
}}

Snapshot JSON to analyse:
{json.dumps(snapshot)}
"""


# ============================================================
# BINGX HELPERS (BEST-EFFORT, FAIL-SAFE)
# ============================================================

def load_supported_bingx_symbols():
    """Try to load which USDT perpetual symbols exist on BingX."""
    global SUPPORTED_BINGX
    if not bingx:
        SUPPORTED_BINGX = set()
        return

    symbols = set()
    try:
        # Try a few possible contract-list endpoints (py-bingx versions differ)
        if hasattr(bingx, "get_all_contracts"):
            data = bingx.get_all_contracts()
            contracts = (
                data.get("data", {}).get("contracts")
                or data.get("contracts")
                or data.get("data")
                or []
            )
        elif hasattr(bingx, "swap_v2_get_contracts"):
            data = bingx.swap_v2_get_contracts()
            contracts = (
                data.get("data", {}).get("contracts")
                or data.get("contracts")
                or data.get("data")
                or []
            )
        else:
            contracts = []

        for c in contracts:
            sym = None
            if isinstance(c, dict):
                sym = c.get("symbol") or c.get("pair")
            elif isinstance(c, str):
                sym = c

            if not sym:
                continue

            # normalize BTC-USDT / BTCUSDT -> BTCUSDT
            if sym.endswith("-USDT"):
                symbols.add(sym.replace("-USDT", "USDT"))
            elif sym.endswith("USDT"):
                symbols.add(sym)

        SUPPORTED_BINGX = symbols
        print(f"[BINGX] Loaded {len(SUPPORTED_BINGX)} symbols")
    except Exception as e:
        print("load_supported_bingx_symbols error:", e)
        SUPPORTED_BINGX = set()


def get_bingx_usdt_balance():
    """Return available USDT margin for futures; None if unknown."""
    if not bingx:
        return None

    try:
        # Try various known methods, fail-safe
        if hasattr(bingx, "swap_v2_get_balance"):
            data = bingx.swap_v2_get_balance()
            balances = (data.get("data") or {}).get("balance") or []
            for b in balances:
                if b.get("asset") == "USDT":
                    return float(b.get("availableBalance") or 0.0)

        elif hasattr(bingx, "get_perpetual_balance"):
            data = bingx.get_perpetual_balance()
            bal = (data.get("data") or {}).get("balance") or {}
            avail = bal.get("availableMargin") or bal.get("balance")
            if avail is not None:
                return float(avail)

        # If we reach here, we don't know; fall back to None
        return None
    except Exception as e:
        print("get_bingx_usdt_balance error:", e)
        return None


async def maybe_auto_trade(signal: dict, app):
    """
    Execute a BingX market order if:
      - BingX client exists
      - symbol supported
      - open positions < AUTO_MAX_POSITIONS
    Uses 3x leverage and splits risk across AUTO_MAX_POSITIONS.
    """
    global auto_open_positions

    if not bingx:
        return

    symbol = signal["symbol"]
    direction = signal["direction"]

    if symbol not in SUPPORTED_BINGX:
        # not tradable on BingX – just skip autotrade
        return

    if len(auto_open_positions) >= AUTO_MAX_POSITIONS:
        if OWNER_CHAT_ID:
            await app.bot.send_message(
                OWNER_CHAT_ID,
                "⚠️ Auto-trade skipped: max auto positions already open.",
            )
        return

    balance = get_bingx_usdt_balance()
    if not balance or balance <= 0:
        if OWNER_CHAT_ID:
            await app.bot.send_message(
                OWNER_CHAT_ID,
                "⚠️ Auto-trade skipped: BingX USDT balance unavailable or zero.",
            )
        return

    entry = signal["entry"]
    if not entry or entry <= 0:
        return

    total_notional = balance * AUTO_LEVERAGE
    qty = (total_notional / AUTO_MAX_POSITIONS) / entry

    bingx_symbol = f"{symbol.replace('USDT', '')}-USDT"
    side = "LONG" if direction.lower() == "long" else "SHORT"

    try:
        if not hasattr(bingx, "open_market_order"):
            print("BingX client has no open_market_order, skipping autotrade.")
            return

        # fire order (py-bingx handles precision server-side)
        bingx.open_market_order(
            bingx_symbol,
            side,
            round(qty, 4),
            tp=str(signal["tp1"]),
            sl=str(signal["sl"]),
        )

        auto_open_positions.add(symbol)

        if OWNER_CHAT_ID:
            msg = (
                "✅ AutoTrade Executed\n"
                f"Symbol: {bingx_symbol}\n"
                f"Side: {side}\n"
                f"Qty: {qty:.4f}\n"
                f"Entry: {entry}\nSL: {signal['sl']}\nTP1: {signal['tp1']}"
            )
            await app.bot.send_message(OWNER_CHAT_ID, msg)
    except Exception as e:
        print("auto_trade error:", e)
        if OWNER_CHAT_ID:
            await app.bot.send_message(
                OWNER_CHAT_ID,
                f"❌ AutoTrade Error: {e}",
            )


# ============================================================
# SNAPSHOT BUILDER FOR MANUAL ANALYSIS
# ============================================================

def build_snapshot(symbol: str, timeframes):
    """
    Returns (snapshot_dict, current_price) using Bybit candles.
    Snapshot: { timeframe: [ {open, high, low, close, volume}, ... ] }
    """
    snapshot = {}
    current_price = None

    for tf in timeframes:
        candles = get_candles(symbol, tf)
        snapshot[tf] = candles
        if candles:
            current_price = candles[-1]["close"]

    if current_price is None:
        current_price = get_price(symbol)

    return snapshot, current_price


# ============================================================
# MANUAL ANALYSIS
# ============================================================

def analyze_manual(symbol: str, timeframe: str | None):
    """
    Manual analysis for commands like:
      /ETHUSDT         (multi TF)
      /ETHUSDT 4h
    """
    # sanitize timeframe
    if timeframe and timeframe not in INTERVAL_MAP:
        timeframe = None

    tfs = [timeframe] if timeframe else MANUAL_TFS_DEFAULT
    snapshot, price = build_snapshot(symbol, tfs)

    if price is None:
        return f"❌ Could not fetch candles for {symbol} from Bybit."

    prompt = manual_prompt(symbol, timeframe, snapshot, price)
    raw = run_gemini(prompt)
    data = extract_json(raw)
    if not data:
        return "❌ Gemini JSON error."

    direction = str(data.get("direction", "flat")).lower()
    up = int(data.get("upside_probability", 0) or 0)
    down = int(data.get("downside_probability", 0) or 0)
    flat = int(data.get("flat_probability", 0) or 0)
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

    # Only show setup if Gemini actually gave trade levels
    entry = data.get("entry")
    sl = data.get("stop")
    tp1 = data.get("tp1")
    tp2 = data.get("tp2")
    rr = data.get("rr")

    # Only show if entry & SL & TP1 exist and dominant prob >= MIN_PROB_SCAN
    dominant = max(up, down)
    if (
        entry is not None
        and sl is not None
        and tp1 is not None
        and dominant >= MIN_PROB_SCAN
    ):
        lines += [
            "",
            f"Entry: `{entry}`",
            f"SL: `{sl}`",
            f"TP1: `{tp1}`",
        ]
        if tp2 is not None:
            lines.append(f"TP2: `{tp2}`")
        if rr is not None:
            lines.append(f"RR: `{rr}`")
    else:
        lines.append("\n⚠️ No high-probability setup (≥ 80%). Trade is optional / avoid.")

    return "\n".join(lines)


# ============================================================
# SCANNER (AUTOSCAN)
# ============================================================

async def scan_once(app):
    global last_signal_time

    if not SCAN_ENABLED or not OWNER_CHAT_ID:
        return

    symbols = await asyncio.to_thread(get_bybit_symbols)
    if not symbols:
        return

    for sym in symbols:
        try:
            c15 = get_candles(sym, "15m")
            c1h = get_candles(sym, "1h")
            c4h = get_candles(sym, "4h")
            price = get_price(sym)
            if price is None:
                continue

            candles = {"15m": c15, "1h": c1h, "4h": c4h}
            prompt = scan_prompt(sym, candles, price)
            raw = await asyncio.to_thread(run_gemini, prompt)
            data = extract_json(raw)
            if not data:
                continue

            direction = str(data.get("direction", "flat")).lower()
            if direction == "flat":
                continue

            prob = int(data.get("probability", 0) or 0)
            rr = float(data.get("rr", 0.0) or 0.0)
            if prob < MIN_PROB_SCAN or rr < MIN_RR:
                continue

            entry = float(data.get("entry") or 0.0)
            sl = float(data.get("stop") or 0.0)
            tp1 = float(data.get("tp1") or 0.0)
            if entry <= 0 or sl <= 0 or tp1 <= 0:
                continue

            key = (sym, direction)
            now = time.time()
            last = last_signal_time.get(key)
            if last and (now - last) < SIGNAL_COOLDOWN_SECONDS:
                continue
            last_signal_time[key] = now

            summary = data.get("summary") or ""

            warn = ""
            if sym not in SUPPORTED_BINGX:
                warn = "ℹ️ This symbol is not supported on BingX. Auto-trade will be skipped.\n\n"

            msg = (
                warn
                + "🚨 *AI SIGNAL*\n"
                f"Symbol: `{sym}`\n"
                f"Direction: `{direction}`\n"
                f"Probability: `{prob}%`\n"
                f"RR: `{rr}`\n"
                f"Entry: `{entry}`\n"
                f"SL: `{sl}`\n"
                f"TP1: `{tp1}`\n"
                f"Reason: _{summary}_"
            )

            await app.bot.send_message(OWNER_CHAT_ID, msg, parse_mode="Markdown")

            # try autotrade (non-blocking)
            sig = {
                "symbol": sym,
                "direction": direction,
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
            }
            await maybe_auto_trade(sig, app)

        except Exception as e:
            print("scan_once error for", sym, ":", e)


async def scan_loop(app):
    await asyncio.sleep(5)
    while True:
        try:
            await scan_once(app)
        except Exception as e:
            print("SCAN LOOP ERR:", e)
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)


# ============================================================
# TELEGRAM HANDLERS
# ============================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SCAN_ENABLED
    SCAN_ENABLED = True
    await update.message.reply_text(
        "✅ Auto scanner ON.\n"
        "• Scans top Bybit USDT futures every 5 minutes.\n"
        f"• Signals only if probability ≥ {MIN_PROB_SCAN}% and RR ≥ {MIN_RR}.\n"
        "• Cooldown per pair/direction: 10 minutes.\n"
        "• Manual analysis: send /BTCUSDT or /ETHUSDT 4h, etc."
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
# POST_INIT & MAIN  (NO asyncio.run HERE)
# ============================================================

async def post_init(app):
    # load BingX tradable symbols once
    await asyncio.to_thread(load_supported_bingx_symbols)
    # start async scanner loop inside PTB event loop
    app.create_task(scan_loop(app))


def main():
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("stop", cmd_stop))
    application.add_handler(MessageHandler(filters.COMMAND, handle_pair))

    print("Bot running with JSON-safe Gemini + scanner...")
    application.run_polling()


if __name__ == "__main__":
    main()

