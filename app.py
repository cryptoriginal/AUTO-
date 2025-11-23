import os
import json
import asyncio
from datetime import datetime, timezone

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
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "0"))

BINGX_API_KEY = os.getenv("BINGX_API_KEY")
BINGX_API_SECRET = os.getenv("BINGX_API_SECRET")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")

# You can override this in Render env as GEMINI_MODEL
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ============================================================
# SETTINGS
# ============================================================

SCAN_ENABLED = True
SCAN_INTERVAL_SECONDS = 300            # 5 minutes
SIGNAL_COOLDOWN_SECONDS = 600          # 10 minutes

MIN_VOLUME = 50_000_000                # 24h quote volume filter
MIN_PROB_MANUAL = 80                   # manual analysis threshold
MIN_PROB_SCAN = 80                     # autoscan threshold
MIN_RR = 1.9                           # minimum RR

DEFAULT_TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "1d"]
SCAN_TIMEFRAMES = ["15m", "1h", "4h"]

AUTO_MAX_POSITIONS = 2
AUTO_LEVERAGE = 3.0

last_signal_time: dict[tuple[str, str], datetime] = {}
auto_open_positions: set[str] = set()
SUPPORTED_BINGX: set[str] = set()      # symbols like "BTCUSDT"

# ============================================================
# CLIENTS
# ============================================================

# Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

# BingX
bingx = None
if BINGX_API_KEY and BINGX_API_SECRET:
    bingx = BingxAPI(BINGX_API_KEY, BINGX_API_SECRET, timestamp="local")


# ============================================================
# SYMBOL HELPERS
# ============================================================

def std_to_bingx(symbol_std: str) -> str:
    """User sends BTCUSDT, BingX expects BTC-USDT."""
    if symbol_std.endswith("USDT"):
        return symbol_std[:-4] + "-USDT"
    return symbol_std


def bingx_to_std(symbol_bx: str) -> str:
    """BingX BTC-USDT -> BTCUSDT."""
    if symbol_bx.endswith("-USDT"):
        return symbol_bx.replace("-USDT", "USDT")
    return symbol_bx


# ============================================================
# LOAD SUPPORTED BINGX SYMBOLS
# ============================================================

def load_supported_bingx_symbols():
    """
    Fill SUPPORTED_BINGX with symbols like 'BTCUSDT'
    using BingX 24h ticker.
    """
    global SUPPORTED_BINGX

    if not bingx:
        SUPPORTED_BINGX = set()
        return

    try:
        data = bingx.market_get_ticker_all()
        lst = data.get("data", [])
        symbols: set[str] = set()
        for item in lst:
            sym_bx = item.get("symbol", "")
            if sym_bx.endswith("-USDT"):
                symbols.add(bingx_to_std(sym_bx))
        SUPPORTED_BINGX = symbols
        print(f"[INIT] Loaded {len(SUPPORTED_BINGX)} BingX USDT-M futures symbols.")
    except Exception as e:
        print("[INIT] load_supported_bingx_symbols error:", e)
        SUPPORTED_BINGX = set()


# ============================================================
# BINGX MARKET DATA
# ============================================================

def fetch_bingx_candles(symbol_std: str, interval: str):
    """
    Uses py-bingx 0.4: swap_v2_get_kline(symbol, interval, limit)
    symbol_std: BTCUSDT
    """
    if not bingx:
        return None

    try:
        sym_bx = std_to_bingx(symbol_std)
        data = bingx.swap_v2_get_kline(symbol=sym_bx, interval=interval, limit=120)
        candles = data.get("data", [])
        if not candles:
            return None

        return [
            {
                "open_time": c["t"],
                "open": float(c["o"]),
                "high": float(c["h"]),
                "low": float(c["l"]),
                "close": float(c["c"]),
                "volume": float(c["v"]),
            }
            for c in candles
        ]
    except Exception as e:
        print("fetch_bingx_candles error:", symbol_std, interval, e)
        return None


def get_klines(symbol_std: str, interval: str):
    """Only BingX candles."""
    return fetch_bingx_candles(symbol_std, interval)


def get_bingx_price(symbol_std: str):
    """Use swap_v2_get_price to get latest price."""
    if not bingx:
        return None
    try:
        sym_bx = std_to_bingx(symbol_std)
        data = bingx.swap_v2_get_price(symbol=sym_bx)
        return float(data.get("data", {}).get("price"))
    except Exception as e:
        print("get_bingx_price error:", symbol_std, e)
        return None


def get_top_bingx_symbols():
    """
    Top BingX symbols by 24h quote volume >= MIN_VOLUME.
    """
    if not bingx:
        return []

    try:
        data = bingx.market_get_ticker_all()
        lst = data.get("data", [])
    except Exception as e:
        print("get_top_bingx_symbols error:", e)
        return []

    filtered = []
    for item in lst:
        sym_bx = item.get("symbol", "")
        if not sym_bx.endswith("-USDT"):
            continue
        sym_std = bingx_to_std(sym_bx)
        quote_vol = float(item.get("quoteVolume", 0) or 0)
        if quote_vol >= MIN_VOLUME:
            filtered.append((sym_std, quote_vol))

    filtered.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in filtered[:25]]


# ============================================================
# SNAPSHOT BUILDER
# ============================================================

def build_snapshot(symbol_std: str, timeframes):
    snapshot = {}
    price = None

    for tf in timeframes:
        candles = get_klines(symbol_std, tf)
        snapshot[tf] = candles
        if candles:
            price = candles[-1]["close"]

    live_price = get_bingx_price(symbol_std)
    if live_price is not None:
        price = live_price

    return snapshot, price


# ============================================================
# GEMINI PROMPTS & HELPERS
# ============================================================

def prompt_manual(symbol, timeframe, snapshot, price):
    return f"""
You are a world-class crypto futures trader.

Use ONLY this JSON format (no explanation text):

{{
  "direction": "long" or "short" or "flat",
  "summary": "1-2 lines why bullish/bearish or avoid trade",
  "upside_probability": 0-100,
  "downside_probability": 0-100,
  "flat_probability": 0-100,
  "entry": float,
  "sl": float,
  "tp1": float,
  "tp2": float,
  "rr": float
}}

Rules:
- Data is only BingX futures OHLCV (multi-timeframe).
- Focus on: VWAP reclaims/rejections, fixed range volume profile
  from recent swing high to swing low, key support/resistance,
  trend structure (HH/HL vs LH/LL), and reversal candles.
- If the market is choppy or unclear, set direction="flat"
  and give high flat_probability.
- If suggesting a trade, SL must be at a logical key level
  (swing high/low or strong S/R) and rr >= 1.9.

Symbol: {symbol}
Requested timeframe: {timeframe}
Current price: {price}
Snapshot (JSON OHLCV): {json.dumps(snapshot)}
"""


def prompt_scan(symbol, snapshot, price):
    return f"""
You are a fast crypto futures scanner.

Return STRICT JSON ONLY:

{{
  "direction": "long" or "short",
  "probability": 0-100,
  "entry": float,
  "sl": float,
  "tp1": float,
  "tp2": float,
  "rr": float
}}

Rules:
- Data is only BingX OHLCV.
- Focus heavily on VWAP + fixed range volume profile,
  key S/R and reversal candles at those levels.
- Only output a signal if probability >= {MIN_PROB_SCAN}
  AND rr >= {MIN_RR}.
- SL must be at a clear key level (swing high/low or major S/R).

Symbol: {symbol}
Current price: {price}
Snapshot (JSON OHLCV): {json.dumps(snapshot)}
"""


def call_gemini(prompt: str):
    try:
        resp = gemini_model.generate_content(prompt)
        return resp.text
    except Exception as e:
        print("call_gemini error:", e)
        return None


def extract_json(text: str):
    if not text:
        return None
    try:
        start = text.index("{")
        end = text.rindex("}")
        js = text[start:end + 1]
        return json.loads(js)
    except Exception as e:
        print("extract_json error:", e)
        print("GEMINI RAW RESPONSE (trimmed):\n", text[:800])
        return None


# ============================================================
# BINGX BALANCE
# ============================================================

def get_bingx_usdt_balance():
    if not bingx:
        return None
    try:
        data = bingx.swap_v2_get_balance()
        balances = data.get("data", {}).get("balance", [])
        for b in balances:
            if b.get("asset") == "USDT":
                return float(b.get("availableBalance", 0))
        return None
    except Exception as e:
        print("get_bingx_usdt_balance error:", e)
        return None


# ============================================================
# AUTO TRADE
# ============================================================

async def auto_trade(sig: dict, bot):
    """
    Execute market order on BingX using 100% balance at AUTO_LEVERAGE,
    split across AUTO_MAX_POSITIONS.
    """
    global auto_open_positions

    if not bingx:
        return

    symbol_std = sig["symbol"]
    if SUPPORTED_BINGX and symbol_std not in SUPPORTED_BINGX:
        return

    if len(auto_open_positions) >= AUTO_MAX_POSITIONS:
        return

    balance = get_bingx_usdt_balance()
    if not balance or balance <= 0:
        await bot.send_message(OWNER_CHAT_ID, "⚠️ AutoTrade skipped: no USDT balance.")
        return

    entry = sig["entry"]
    if entry <= 0:
        return

    total_notional = balance * AUTO_LEVERAGE
    qty = (total_notional / AUTO_MAX_POSITIONS) / entry

    side = "LONG" if sig["direction"] == "long" else "SHORT"
    symbol_bx = std_to_bingx(symbol_std)

    try:
        _order = bingx.open_market_order(
            symbol_bx,
            side,
            f"{qty:.6f}",
            tp=str(sig["tp1"]),
            sl=str(sig["sl"]),
        )

        auto_open_positions.add(symbol_std)

        await bot.send_message(
            OWNER_CHAT_ID,
            f"✅ AutoTrade Executed\n"
            f"Symbol: {symbol_bx}\nSide: {side}\nQty: {qty:.6f}\n"
            f"Entry: {entry}\nSL: {sig['sl']}\nTP1: {sig['tp1']}"
        )
    except Exception as e:
        await bot.send_message(OWNER_CHAT_ID, f"❌ AutoTrade Error: {e}")


# ============================================================
# MANUAL ANALYSIS
# ============================================================

async def analyze_manual(symbol_std: str, timeframe: str | None):
    tfs = [timeframe] if timeframe else DEFAULT_TIMEFRAMES
    snapshot, price = build_snapshot(symbol_std, tfs)

    if price is None:
        return f"❌ Could not fetch candles for {symbol_std} from BingX."

    prompt = prompt_manual(symbol_std, timeframe, snapshot, price)
    raw = call_gemini(prompt)
    data = extract_json(raw)

    if not data:
        return "❌ Gemini JSON error."

    up = int(data.get("upside_probability", 0) or 0)
    down = int(data.get("downside_probability", 0) or 0)
    flat = int(data.get("flat_probability", 0) or 0)
    summary = data.get("summary", "").strip()
    direction = (data.get("direction") or "").lower()
    rr = float(data.get("rr", 0) or 0)

    main_prob = max(up, down)

    lines = [
        f"📊 *{symbol_std} Analysis*",
        f"Price: `{price}`",
        f"Direction: *{direction}*",
        f"Upside: `{up}%`",
        f"Downside: `{down}%`",
        f"Flat: `{flat}%`",
    ]

    if summary:
        lines.append("")
        lines.append(f"💡 *Why:* {summary}")

    if (
        main_prob >= MIN_PROB_MANUAL
        and direction in ("long", "short")
        and data.get("entry") is not None
    ):
        lines.extend(
            [
                "",
                f"Entry: `{data['entry']}`",
                f"SL: `{data['sl']}`",
                f"TP1: `{data['tp1']}`",
                f"TP2: `{data['tp2']}`",
                f"RR: `{rr}`",
            ]
        )
    else:
        lines.append("")
        lines.append(
            f"⛔ Probability below {MIN_PROB_MANUAL}% for a clean setup, "
            f"so no entry/SL/TP suggested."
        )

    if SUPPORTED_BINGX and symbol_std not in SUPPORTED_BINGX:
        lines.append("")
        lines.append(
            "ℹ️ This pair is not available on BingX USDT-M futures, "
            "so auto-trade is disabled for it."
        )

    return "\n".join(lines)


# ============================================================
# TELEGRAM HANDLERS
# ============================================================

async def handle_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message.text.strip()
    parts = msg.split()

    symbol_std = parts[0].replace("/", "").upper()     # /ethusdt -> ETHUSDT
    timeframe = parts[1] if len(parts) > 1 else None

    await update.message.reply_text(f"⏳ Analysing {symbol_std} ...")

    try:
        result = await analyze_manual(symbol_std, timeframe)
    except Exception as e:
        result = f"❌ Error analysing {symbol_std}: {e}"

    await update.message.reply_markdown(result)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SCAN_ENABLED
    SCAN_ENABLED = True
    await update.message.reply_text(
        "✅ Auto Scanner ON (BingX only, every 5 min, 10 min cooldown per symbol/direction)."
    )


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SCAN_ENABLED
    SCAN_ENABLED = False
    await update.message.reply_text("⏹ Auto Scanner OFF")


# ============================================================
# AUTOSCAN
# ============================================================

def analyze_scan_sync(symbol_std: str):
    snapshot, price = build_snapshot(symbol_std, SCAN_TIMEFRAMES)
    if price is None:
        return None

    prompt = prompt_scan(symbol_std, snapshot, price)
    raw = call_gemini(prompt)
    data = extract_json(raw)
    if not data:
        return None

    prob = int(data.get("probability", 0) or 0)
    rr = float(data.get("rr", 0) or 0)

    if prob < MIN_PROB_SCAN or rr < MIN_RR:
        return None

    return {
        "symbol": symbol_std,
        "direction": (data.get("direction") or "").lower(),
        "probability": prob,
        "entry": float(data["entry"]),
        "sl": float(data["sl"]),
        "tp1": float(data["tp1"]),
        "tp2": float(data.get("tp2", data["tp1"])),
        "rr": rr,
    }


async def run_scan_once(app):
    global last_signal_time

    if not SCAN_ENABLED or OWNER_CHAT_ID == 0:
        return

    try:
        symbols = get_top_bingx_symbols()
    except Exception as e:
        print("run_scan_once get_top_bingx_symbols error:", e)
        return

    now = datetime.now(timezone.utc)

    for sym in symbols:
        try:
            sig = analyze_scan_sync(sym)
        except Exception as e:
            print("analyze_scan_sync error:", sym, e)
            continue

        if not sig:
            continue

        key = (sym, sig["direction"])
        last = last_signal_time.get(key)
        if last and (now - last).total_seconds() < SIGNAL_COOLDOWN_SECONDS:
            continue

        last_signal_time[key] = now

        msg_lines = [
            f"🚨 *AI SIGNAL*",
            f"Symbol: `{sym}`",
            f"Direction: `{sig['direction']}`",
            f"Probability: `{sig['probability']}%`",
            f"RR: `{sig['rr']}`",
            f"Entry: `{sig['entry']}`",
            f"SL: `{sig['sl']}`",
            f"TP1: `{sig['tp1']}`",
        ]

        await app.bot.send_message(
            OWNER_CHAT_ID,
            "\n".join(msg_lines),
            parse_mode="Markdown",
        )

        await auto_trade(sig, app.bot)


async def scanner_loop(app):
    while True:
        try:
            await run_scan_once(app)
        except Exception as e:
            print("scanner_loop error:", e)
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)


async def post_init(app):
    asyncio.create_task(scanner_loop(app))


# ============================================================
# MAIN
# ============================================================

def main():
    load_supported_bingx_symbols()

    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("stop", cmd_stop))

    coin_command_filter = filters.COMMAND & ~filters.Regex(r"^/(start|stop)$")
    application.add_handler(MessageHandler(coin_command_filter, handle_pair))

    application.run_polling()


if __name__ == "__main__":
    main()
