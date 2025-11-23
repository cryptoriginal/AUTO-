import os
import json
import time
from datetime import datetime, timezone

import requests
from google import genai
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    CallbackContext,
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

GEMINI_MODEL_MANUAL = "gemini-2.5-flash"
GEMINI_MODEL_SCANNER = "gemini-2.5-flash"


# ============================================================
# SCANNER SETTINGS
# ============================================================

SCAN_ENABLED = True
SCAN_INTERVAL_SECONDS = 300  # 5 minutes

MIN_VOLUME = 50_000_000
MAX_SCAN_SYMBOLS = 25

MIN_PROB_MANUAL = 75
MIN_PROB_SCAN = 85
MIN_RR = 2.1

DEFAULT_TIMEFRAMES = [
    "5m", "15m", "30m", "1h", "2h", "4h",
    "6h", "12h", "1d", "1w"
]
SCAN_TIMEFRAMES = ["15m", "1h", "4h"]

AUTO_MAX_POSITIONS = 2
AUTO_LEVERAGE = 3.0

last_signal_time = {}
auto_open_positions = set()

# None = we failed to load list → treat all as “maybe supported”
# set([...]) = list successfully loaded → we check membership
SUPPORTED_BINGX = None


# ============================================================
# CLIENTS
# ============================================================

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

bingx = None
if BINGX_API_KEY and BINGX_API_SECRET:
    bingx = BingxAPI(BINGX_API_KEY, BINGX_API_SECRET, timestamp="local")


# ============================================================
# LOAD SUPPORTED BINGX FUTURES SYMBOLS
# ============================================================

def load_supported_bingx_symbols():
    """
    Load all USDT-M perpetual futures from BingX.

    If this fails, we set SUPPORTED_BINGX = None
    which means: do NOT warn about unsupported, just TRY trading and
    let BingX respond if symbol really doesn't exist.
    """
    global SUPPORTED_BINGX

    if not bingx:
        print("No BingX client; skipping contract load.")
        SUPPORTED_BINGX = None
        return

    try:
        data = bingx.swap_v2_get_contracts()
        contracts = data.get("data", {}).get("contracts", [])
        symbols = set()

        for c in contracts:
            sym = c.get("symbol", "")
            # Typical futures format: BTC-USDT
            if sym.endswith("-USDT"):
                symbols.add(sym.replace("-USDT", "USDT"))

        SUPPORTED_BINGX = symbols
        print(f"Loaded {len(SUPPORTED_BINGX)} BingX USDT-M futures symbols.")
    except Exception as e:
        print("Failed to load BingX contracts:", e)
        SUPPORTED_BINGX = None  # no info, no warnings


# ============================================================
# MARKET DATA (TRIPLE FALLBACK)
# ============================================================

BINANCE_ENDPOINTS = [
    "https://fapi.binancevip.com",
    "https://api2.binance.com"
]

OKX_ENDPOINT = "https://www.okx.com"


def fetch_binance(path, params=None):
    """
    Binance with retries. Used for klines and 24h ticker.
    """
    params = params or {}
    for endpoint in BINANCE_ENDPOINTS:
        url = endpoint + path

        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=10)

                if r.status_code in [418, 429, 451, 403] or r.status_code >= 500:
                    time.sleep(1.2 * (attempt + 1))
                    continue

                r.raise_for_status()
                return r.json()
            except Exception:
                time.sleep(1.2 * (attempt + 1))

    raise RuntimeError("Binance failed in all attempts.")


def fetch_okx(symbol, interval):
    """
    OKX futures candles: ETHUSDT → ETH-USDT-SWAP
    """
    try:
        okx_symbol = symbol.replace("USDT", "-USDT-SWAP")
        url = f"{OKX_ENDPOINT}/api/v5/market/candles"
        r = requests.get(url, params={"instId": okx_symbol, "bar": interval}, timeout=10)
        r.raise_for_status()
        raw = r.json()

        candles = [{
            "open_time": c[0],
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]),
        } for c in raw.get("data", [])]

        return candles if candles else None
    except Exception as e:
        print(f"fetch_okx error {symbol} {interval}:", e)
        return None


def fetch_bingx(symbol, interval):
    """
    BingX futures candles: ETHUSDT → ETH-USDT
    """
    if not bingx:
        return None

    try:
        bingx_symbol = symbol.replace("USDT", "-USDT")
        data = bingx.market_get_candles(bingx_symbol, interval, 100)

        candles = [{
            "open_time": c["t"],
            "open": float(c["o"]),
            "high": float(c["h"]),
            "low": float(c["l"]),
            "close": float(c["c"]),
            "volume": float(c["v"]),
        } for c in data.get("data", [])]

        return candles if candles else None
    except Exception as e:
        print(f"fetch_bingx error {symbol} {interval}:", e)
        return None


def get_klines(symbol, interval):
    """
    Master candle function:
    1) BingX
    2) OKX
    3) Binance
    """
    # 1️⃣ BingX
    c = fetch_bingx(symbol, interval)
    if c:
        return c

    # 2️⃣ OKX
    c = fetch_okx(symbol, interval)
    if c:
        return c

    # 3️⃣ Binance
    try:
        raw = fetch_binance("/fapi/v1/klines", {
            "symbol": symbol,
            "interval": interval,
            "limit": 100,
        })
    except Exception as e:
        print(f"fetch_binance klines error {symbol} {interval}:", e)
        return None

    candles = [{
        "open_time": c[0],
        "open": float(c[1]),
        "high": float(c[2]),
        "low": float(c[3]),
        "close": float(c[4]),
        "volume": float(c[5]),
    } for c in raw]

    return candles if candles else None


# ============================================================
# TOP SYMBOLS FOR SCANNER (BINANCE 24H VOL)
# ============================================================

def get_top_symbols():
    """
    Use Binance 24h ticker just to pick high-volume pairs.
    Scanner & trades still use BingX/OKX/Binance candles above.
    """
    try:
        data = fetch_binance("/fapi/v1/ticker/24hr")
    except Exception as e:
        print("get_top_symbols: Binance 24h error:", e)
        return []

    pairs = [
        s for s in data
        if s.get("symbol", "").endswith("USDT")
        and float(s.get("quoteVolume", 0)) >= MIN_VOLUME
    ]

    pairs.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    symbols = [p["symbol"] for p in pairs[:MAX_SCAN_SYMBOLS]]

    # If we successfully loaded a BingX list, filter to those for scanning
    if SUPPORTED_BINGX is not None:
        symbols = [s for s in symbols if s in SUPPORTED_BINGX]

    return symbols


# ============================================================
# SNAPSHOT BUILDER
# ============================================================

def build_snapshot(symbol, timeframes):
    snapshot = {}
    current_price = None

    for tf in timeframes:
        try:
            klines = get_klines(symbol, tf)
        except Exception as e:
            print(f"get_klines error {symbol} {tf}:", e)
            klines = None

        snapshot[tf] = klines
        if klines:
            current_price = klines[-1]["close"]

    return snapshot, current_price


# ============================================================
# GEMINI PROMPTS
# ============================================================

def prompt_for_pair(symbol, timeframe, snapshot, price):
    return f"""
You are a world-class crypto futures analyst with deep knowledge of:
- Price action, liquidity, key levels, trendlines, chart patterns
- RSI / MACD / EMAs as confirmation
- Reversal candles (hammer, doji, engulfing, pin-bars)

You analyse the given symbol using the multi-timeframe snapshot
and output strict JSON with:

- upside_probability
- downside_probability
- flat_probability
- direction  ("long" | "short" | "flat")
- entry
- sl
- tp1
- tp2
- rr   (risk:reward)

Rules:
- rr must be >= 1.8 if you propose a trade.
- SL must be at a real key level / liquidity area, not random.
- Use price action first, indicators only as confirmation.
- If market is choppy / unclear, increase flat_probability.

symbol: {symbol}
requested_timeframe: {timeframe}
current_price: {price}
snapshot: {json.dumps(snapshot)}
Return ONLY JSON, no explanation.
"""


def prompt_for_scan(symbol, snapshot, price):
    return f"""
You are an ultra-fast crypto futures scanner.

You scan the symbol and decide if there is a high probability trade
(upside or downside) based on this snapshot.

Return strict JSON with:
- direction  ("long" | "short")
- probability  (0-100)
- entry
- sl
- tp1
- tp2
- rr  (risk:reward)

Rules:
- Only give a trade if probability >= {MIN_PROB_SCAN} and rr >= {MIN_RR}.
- SL must be at a logical key level, not random.
- Entry can be near current price or slight pullback.
- Prefer clean RR structure.

symbol: {symbol}
current_price: {price}
snapshot: {json.dumps(snapshot)}
Return ONLY JSON.
"""


# ============================================================
# GEMINI HELPERS
# ============================================================

def call_gemini(prompt, model):
    r = gemini_client.models.generate_content(
        model=model,
        contents=prompt
    )
    return r.text


def extract_json(text):
    try:
        start = text.index("{")
        end = text.rindex("}")
        return json.loads(text[start:end+1])
    except Exception:
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
        print("BalanceError:", e)
        return None


# ============================================================
# AUTO-TRADE
# ============================================================

def binance_to_bingx_symbol(symbol: str) -> str:
    return symbol.replace("USDT", "-USDT")


def maybe_auto_trade(sig, context: CallbackContext):
    global auto_open_positions

    if not bingx:
        return

    symbol = sig["symbol"]

    # Only warn if we actually have a contract list
    if SUPPORTED_BINGX is not None and symbol not in SUPPORTED_BINGX:
        context.bot.send_message(
            OWNER_CHAT_ID,
            f"ℹ️ {symbol} not found in BingX futures contracts list.\n"
            f"Analysis only, auto-trade disabled."
        )
        return

    if len(auto_open_positions) >= AUTO_MAX_POSITIONS:
        return

    balance = get_bingx_usdt_balance()
    if not balance or balance <= 0:
        context.bot.send_message(
            OWNER_CHAT_ID,
            "⚠️ Auto-trade skipped: USDT balance unavailable."
        )
        return

    entry = sig["entry"]
    if entry <= 0:
        return

    total_notional = balance * AUTO_LEVERAGE
    qty = (total_notional / AUTO_MAX_POSITIONS) / entry

    bingx_symbol = binance_to_bingx_symbol(symbol)
    side = "LONG" if sig["direction"] == "long" else "SHORT"

    try:
        order = bingx.open_market_order(
            bingx_symbol,
            side,
            f"{qty:.6f}",
            tp=str(sig["tp1"]),
            sl=str(sig["sl"])
        )

        auto_open_positions.add(symbol)

        context.bot.send_message(
            OWNER_CHAT_ID,
            f"✅ AutoTrade Executed\n"
            f"Symbol: {bingx_symbol}\n"
            f"Side: {side}\n"
            f"Qty: {qty:.6f}\n"
            f"Entry: {entry}\n"
            f"SL: {sig['sl']}\n"
            f"TP1: {sig['tp1']}"
        )

    except Exception as e:
        context.bot.send_message(OWNER_CHAT_ID, f"❌ AutoTrade Error: {e}")


# ============================================================
# ANALYSIS & SCANNER
# ============================================================

def analyze_command(symbol, timeframe):
    tfs = [timeframe] if timeframe else DEFAULT_TIMEFRAMES

    snapshot, price = build_snapshot(symbol, tfs)
    if price is None:
        return f"❌ Could not fetch candles for {symbol} from any exchange."

    raw = call_gemini(
        prompt_for_pair(symbol, timeframe, snapshot, price),
        GEMINI_MODEL_MANUAL,
    )
    data = extract_json(raw)

    if not data:
        return "❌ JSON parsing error from AI.\n" + raw[:500]

    warn = ""
    if SUPPORTED_BINGX is not None and symbol not in SUPPORTED_BINGX:
        warn = (
            f"ℹ️ {symbol} not in BingX futures contracts list.\n"
            f"Analysis only, auto-trade disabled.\n\n"
        )

    result = [
        warn + f"📊 *{symbol} Analysis*",
        f"Price: `{price}`",
        f"Direction: *{data.get('direction')}*",
        f"Upside: `{data.get('upside_probability')}%`",
        f"Downside: `{data.get('downside_probability')}%`",
        f"Flat: `{data.get('flat_probability')}%`",
    ]

    if data.get("entry"):
        result.extend([
            "",
            f"Entry: `{data['entry']}`",
            f"SL: `{data['sl']}`",
            f"TP1: `{data['tp1']}`",
            f"TP2: `{data.get('tp2')}`",
        ])

    return "\n".join(result)


def analyze_scan(symbol):
    snapshot, price = build_snapshot(symbol, SCAN_TIMEFRAMES)
    if price is None:
        return None

    raw = call_gemini(
        prompt_for_scan(symbol, snapshot, price),
        GEMINI_MODEL_SCANNER,
    )
    data = extract_json(raw)

    if not data:
        return None

    prob = int(data.get("probability", 0))
    rr = float(data.get("rr", 0))

    if prob < MIN_PROB_SCAN or rr < MIN_RR:
        return None

    return {
        "symbol": symbol,
        "direction": data["direction"],
        "probability": prob,
        "entry": float(data["entry"]),
        "sl": float(data["sl"]),
        "tp1": float(data["tp1"]),
        "tp2": float(data.get("tp2", data["tp1"])),
        "rr": rr,
    }


# ============================================================
# TELEGRAM HANDLERS
# ============================================================

def cmd_start(update, context: CallbackContext):
    global SCAN_ENABLED
    SCAN_ENABLED = True
    update.message.reply_text("✅ Auto Scanner ON")


def cmd_stop(update, context: CallbackContext):
    global SCAN_ENABLED
    SCAN_ENABLED = False
    update.message.reply_text("⏹ Auto Scanner OFF")


def handle_pair(update, context: CallbackContext):
    msg = update.message.text.strip()
    parts = msg.split()

    symbol = parts[0].replace("/", "").upper()
    timeframe = parts[1] if len(parts) > 1 else None

    update.message.reply_text(f"⏳ Analysing {symbol}...")

    try:
        result = analyze_command(symbol, timeframe)
    except Exception as e:
        result = f"❌ Error analysing {symbol}: {e}"

    update.message.reply_markdown(result)


# ============================================================
# SCANNER JOB
# ============================================================

def scanner_job(context: CallbackContext):
    if not SCAN_ENABLED:
        return
    if OWNER_CHAT_ID == 0:
        return

    try:
        symbols = get_top_symbols()
    except Exception as e:
        print("scanner_job get_top_symbols error:", e)
        return

    now = datetime.now(timezone.utc)

    for sym in symbols:
        try:
            sig = analyze_scan(sym)
        except Exception as e:
            print("scanner_job analyze_scan error:", sym, e)
            continue

        if not sig:
            continue

        key = (sym, sig["direction"])
        last = last_signal_time.get(key)
        if last and (now - last).total_seconds() < 1800:
            continue  # 30m cooldown

        last_signal_time[key] = now

        warn = ""
        if SUPPORTED_BINGX is not None and sym not in SUPPORTED_BINGX:
            warn = (
                f"ℹ️ {sym} not in BingX futures contracts list.\n"
                f"Analysis only, auto-trade disabled.\n\n"
            )

        context.bot.send_message(
            OWNER_CHAT_ID,
            warn +
            f"🚨 *AI SIGNAL*\n"
            f"Symbol: `{sym}`\n"
            f"Direction: `{sig['direction']}`\n"
            f"Probability: `{sig['probability']}%`\n"
            f"RR: `{sig['rr']}`\n"
            f"Entry: `{sig['entry']}`\n"
            f"SL: `{sig['sl']}`\n"
            f"TP1: `{sig['tp1']}`",
            parse_mode="Markdown",
        )

        maybe_auto_trade(sig, context)


# ============================================================
# MAIN
# ============================================================

def main():
    load_supported_bingx_symbols()

    updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", cmd_start))
    dp.add_handler(CommandHandler("stop", cmd_stop))
    dp.add_handler(MessageHandler(Filters.command, handle_pair))

    jq = updater.job_queue
    jq.run_repeating(scanner_job, interval=SCAN_INTERVAL_SECONDS, first=10)

    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
