import os
import json
import time
from datetime import datetime, timezone

import requests
from google import genai
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
# ENVIRONMENT VARIABLES
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "0"))

BINGX_API_KEY = os.getenv("BINGX_API_KEY")
BINGX_API_SECRET = os.getenv("BINGX_API_SECRET")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN missing")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing")

GEMINI_MODEL_MANUAL = "gemini-2.5-flash"
GEMINI_MODEL_SCANNER = "gemini-2.5-flash"  # you can later change to cheaper model

# ============================================================
# STRATEGY SETTINGS
# ============================================================

SCAN_ENABLED = True                      # controlled by /start and /stop
SCAN_INTERVAL_SECONDS = 300              # 5 min
MIN_VOLUME = 50_000_000                  # 24h quote volume filter
MAX_SCAN_SYMBOLS = 25

MIN_PROB_MANUAL = 75                     # for manual analysis messages
MIN_PROB_SCAN = 85                       # for auto-scan + auto-trade
MIN_RR = 2.1                             # min RR for both manual & auto

DEFAULT_TIMEFRAMES = [
    "5m", "15m", "30m", "1h", "2h", "4h",
    "6h", "12h", "1d", "1w",
]
SCAN_TIMEFRAMES = ["15m", "1h", "4h"]

AUTO_MAX_POSITIONS = 2
AUTO_LEVERAGE = 3.0                      # use 100% balance at 3x, split across 2 slots

last_signal_time = {}                    # (symbol, direction) -> datetime
auto_open_positions = set()             # track which symbols we auto-opened

# ============================================================
# API ENDPOINTS
# ============================================================

BINGX_BASE = "https://open-api.bingx.com"
OKX_BASE = "https://www.okx.com"
BINANCE_ENDPOINTS = [
    "https://fapi.binance.com",
    "https://api2.binance.com",
]

# ============================================================
# CLIENTS
# ============================================================

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

bingx = None
if BINGX_API_KEY and BINGX_API_SECRET:
    bingx = BingxAPI(BINGX_API_KEY, BINGX_API_SECRET, timestamp="local")

# ============================================================
# SYMBOL HELPERS
# ============================================================

def binance_to_bingx_symbol(symbol: str) -> str:
    """
    Binance style: BTCUSDT, SUIUSDT
    BingX swap style: BTC-USDT, SUI-USDT
    """
    if symbol.endswith("USDT"):
        return symbol[:-4] + "-USDT"
    return symbol


def bingx_to_binance_symbol(symbol: str) -> str:
    """
    BingX: BTC-USDT -> Binance: BTCUSDT
    """
    return symbol.replace("-", "")


def binance_to_okx_inst(symbol: str) -> str:
    """
    Binance: BTCUSDT -> OKX swap instId: BTC-USDT-SWAP
    """
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        return f"{base}-USDT-SWAP"
    return symbol + "-USDT-SWAP"


OKX_BAR_MAP = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "12h": "12H",
    "1d": "1D",
    "1w": "1W",
}

# ============================================================
# BINANCE REQUEST (FALLBACK ONLY)
# ============================================================

def binance_request(path, params=None):
    params = params or {}
    last_error = None

    for endpoint in BINANCE_ENDPOINTS:
        url = endpoint + path
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=10)

                # handle 418/429/451/403 & 5xx
                if r.status_code in [418, 429, 451, 403] or r.status_code >= 500:
                    last_error = f"Binance status {r.status_code}"
                    time.sleep(1.2 * (attempt + 1))
                    continue

                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_error = str(e)
                time.sleep(1.2 * (attempt + 1))

    raise RuntimeError(f"Binance error: {last_error or 'unknown'}")

# ============================================================
# KLINE FETCHERS (TRIPLE FALLBACK: BINGX -> OKX -> BINANCE)
# ============================================================

def fetch_bingx_klines(symbol: str, interval: str, limit: int):
    sym = binance_to_bingx_symbol(symbol)
    url = BINGX_BASE + "/openApi/swap/v3/quote/klines"
    params = {
        "symbol": sym,
        "interval": interval,
        "limit": str(limit),
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    rows = data.get("data") if isinstance(data, dict) else data
    if not rows:
        raise RuntimeError("No BingX kline data")

    # BingX format: [openTime, open, high, low, close, volume, ...]
    klines = []
    for it in rows:
        # ensure at least 6 fields
        if len(it) < 6:
            continue
        open_time = int(it[0])
        open_p = str(it[1])
        high_p = str(it[2])
        low_p = str(it[3])
        close_p = str(it[4])
        vol = str(it[5])
        klines.append([open_time, open_p, high_p, low_p, close_p, vol])

    if not klines:
        raise RuntimeError("Empty BingX klines after parsing")
    return klines


def fetch_okx_klines(symbol: str, interval: str, limit: int):
    bar = OKX_BAR_MAP.get(interval)
    if not bar:
        raise RuntimeError(f"OKX does not support bar {interval}")

    inst_id = binance_to_okx_inst(symbol)
    url = OKX_BASE + "/api/v5/market/candles"
    params = {
        "instId": inst_id,
        "bar": bar,
        "limit": str(limit),
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    rows = data.get("data") if isinstance(data, dict) else data
    if not rows:
        raise RuntimeError("No OKX kline data")

    # OKX format: [ts, o, h, l, c, vol, volCcy, volCcyQuote, ...]
    klines = []
    # OKX returns newest first; reverse for chronological
    for it in reversed(rows):
        if len(it) < 6:
            continue
        ts = int(it[0])
        o = str(it[1])
        h = str(it[2])
        l = str(it[3])
        c = str(it[4])
        vol = str(it[5])
        klines.append([ts, o, h, l, c, vol])

    if not klines:
        raise RuntimeError("Empty OKX klines after parsing")
    return klines


def fetch_binance_klines(symbol: str, interval: str, limit: int):
    return binance_request(
        "/fapi/v1/klines",
        {"symbol": symbol, "interval": interval, "limit": limit},
    )


def get_klines(symbol: str, interval: str, limit: int = 120):
    """
    Triple fallback order:
    1) BingX
    2) OKX
    3) Binance
    Returns a list of [open_time, open, high, low, close, volume]
    """
    # 1. BingX
    try:
        return fetch_bingx_klines(symbol, interval, limit)
    except Exception:
        pass

    # 2. OKX
    try:
        return fetch_okx_klines(symbol, interval, limit)
    except Exception:
        pass

    # 3. Binance
    try:
        return fetch_binance_klines(symbol, interval, limit)
    except Exception as e:
        raise RuntimeError("All venues failed for klines") from e

# ============================================================
# TOP SYMBOLS (BINGX PRIMARY, BINANCE FALLBACK)
# ============================================================

def get_top_symbols():
    """
    Returns list of Binance-style symbols (e.g. BTCUSDT)
    filtered by MIN_VOLUME 24h quote volume.
    """
    # --- Primary: BingX 24h ticker ---
    try:
        url = BINGX_BASE + "/openApi/swap/v2/quote/ticker"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        rows = data.get("data") if isinstance(data, dict) else data

        candidates = []
        for item in rows or []:
            sym_bx = item.get("symbol", "")
            if not sym_bx.endswith("-USDT"):
                continue
            vol = float(
                item.get("quoteVolume")
                or item.get("quoteVolume24h")
                or item.get("quoteQty")
                or 0.0
            )
            if vol < MIN_VOLUME:
                continue
            sym_binance = bingx_to_binance_symbol(sym_bx)
            candidates.append((sym_binance, vol))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [c[0] for c in candidates[:MAX_SCAN_SYMBOLS]]
    except Exception:
        pass

    # --- Fallback: Binance 24h ticker ---
    try:
        data = binance_request("/fapi/v1/ticker/24hr")
        pairs = [
            s for s in data
            if s.get("symbol", "").endswith("USDT")
            and float(s.get("quoteVolume", 0.0)) >= MIN_VOLUME
        ]
        pairs.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
        return [p["symbol"] for p in pairs[:MAX_SCAN_SYMBOLS]]
    except Exception:
        # final fallback: hardcoded majors
        return ["BTCUSDT", "ETHUSDT", "SUIUSDT", "SOLUSDT", "BNBUSDT"]

# ============================================================
# SNAPSHOT BUILDER
# ============================================================

def build_snapshot(symbol, timeframes):
    snapshot = {}
    current_price = None

    for tf in timeframes:
        kl = get_klines(symbol, tf, 100)
        candles = []
        for c in kl:
            candles.append(
                {
                    "open_time": int(c[0]),
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5]),
                }
            )
        snapshot[tf] = candles
        if candles:
            current_price = candles[-1]["close"]

    return snapshot, current_price

# ============================================================
# GEMINI PROMPTS
# ============================================================

def prompt_for_pair(symbol, timeframe, snapshot, price):
    return f"""
You are an elite crypto futures trader and technical analyst.

Your job is to analyse this PERPETUAL futures pair and output a very clear,
trading-desk-style view.

Symbol: {symbol}
Current price: {price}
Focus timeframe: {timeframe if timeframe else "Multi-timeframe view (5m..1w)"}

You are given multi-timeframe OHLCV JSON for this pair:
{json.dumps(snapshot)[:80000]}

RULES FOR ANALYSIS
------------------
1. Use PRICE ACTION first:
   - Trend direction on each timeframe (up / down / range).
   - Major support & resistance, supply & demand zones.
   - Breakouts, retests, trendlines, channels.
   - Reversal candles at key levels (hammer, shooting star, engulfing, pin bar, doji).
   - Classical patterns (triangles, flags, wedges, head-and-shoulders, double top/bottom).

2. Use a FIXED RANGE VOLUME PROFILE mental model:
   - Imagine the main high-volume nodes and low-volume nodes over the recent range.
   - Stop loss must be placed BEYOND a logical invalidation point (beyond swing high/low
     or beyond a high-volume node).
   - Do NOT put SL at random levels.

3. Indicators are only confirmation:
   - EMAs 20 / 50 / 200 for trend & dynamic S/R.
   - RSI for overbought/oversold and divergences.
   - MACD for momentum & trend confirmation.
   - Volume spikes for breakout confirmation.

4. Trade quality filters:
   - Only accept a setup if realistic Risk:Reward rr_ratio >= {MIN_RR}.
   - Upside/downside probability must match the RR and distance to TP.
   - If market is choppy/flat, avoid forcing a trade.

TASK
----
1. Produce probabilities (0-100) for the next meaningful move:
   - upside
   - downside
   - flat (range / chop)

2. Decide best_direction = upside / downside / flat.

3. ONLY IF:
   - best_direction is upside or downside, AND
   - its probability >= {MIN_PROB_MANUAL}, AND
   - a clean setup with rr_ratio >= {MIN_RR} exists,
   THEN give a trade_plan with:
   - direction (long / short)
   - entry (good entry zone near current price)
   - stop_loss (logical invalidation level, based on key level & volume profile idea)
   - take_profit_1 (sensible, realistic TP)
   - take_profit_2 (more ambitious TP, but still reasonably reachable)
   - rr_ratio
   - leverage_hint (e.g. 2-4x, 5-8x, etc., but never insane)
   - confidence (0-100)
   - reasoning (short but clear explanation in trader language)

4. If there is NO clean high-quality setup, set direction="none" and rr_ratio=0.

Return ONLY strict JSON in this exact schema:

{{
  "symbol": "{symbol}",
  "probabilities": {{
    "upside": 0,
    "downside": 0,
    "flat": 0
  }},
  "best_direction": "upside | downside | flat",
  "overall_view": "text summary of market structure",
  "trade_plan": {{
    "direction": "long | short | none",
    "entry": 0,
    "stop_loss": 0,
    "take_profit_1": 0,
    "take_profit_2": 0,
    "rr_ratio": 0,
    "leverage_hint": "",
    "confidence": 0,
    "reasoning": "short explanation"
  }}
}}
"""


def prompt_for_scan(symbol, snapshot, price):
    return f"""
You are an ultra-fast crypto futures scanner.

Symbol: {symbol}
Current price: {price}

You get multi-timeframe OHLCV JSON:
{json.dumps(snapshot)[:60000]}

GOAL
----
Find ONLY very high probability moves that are worth auto-trading.

RULES
-----
1. Focus on 15m / 1h / 4h structure.
2. Combine:
   - Trend direction
   - Key support/resistance
   - Breakouts / breakdowns / retests
   - Clear reversal signals at levels (hammer, engulfing, pin bar)
3. Think in terms of fixed-range volume profile:
   - SL must be outside a clear invalidation zone (beyond swing or volume cluster).
4. STRICT FILTERS:
   - best_direction must be upside or downside (NOT flat)
   - probability(best_direction) >= {MIN_PROB_SCAN} %
   - rr_ratio >= {MIN_RR} (1:{MIN_RR} or better)
5. The TP1 level must be realistic given the probability and market structure,
   not a moonshot.

OUTPUT
------
Return ONLY strict JSON:

{{
  "symbol": "{symbol}",
  "probabilities": {{
    "upside": 0,
    "downside": 0,
    "flat": 0
  }},
  "best_direction": "upside | downside | flat",
  "trade_plan": {{
    "direction": "long | short | none",
    "entry": 0,
    "stop_loss": 0,
    "take_profit_1": 0,
    "take_profit_2": 0,
    "rr_ratio": 0,
    "confidence": 0
  }}
}}

If there is no clean high-quality setup, set direction="none".
"""

# ============================================================
# GEMINI UTILS
# ============================================================

def call_gemini(prompt: str, model: str) -> str:
    resp = gemini_client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return resp.text


def extract_json(text: str):
    try:
        start = text.index("{")
        end = text.rindex("}")
        return json.loads(text[start:end + 1])
    except Exception:
        return None

# ============================================================
# MANUAL ANALYSIS (CHAT COMMAND)
# ============================================================

def analyze_command(symbol: str, timeframe: str | None):
    tfs = [timeframe] if timeframe else DEFAULT_TIMEFRAMES
    snapshot, price = build_snapshot(symbol, tfs)

    if price is None:
        return f"❌ Could not fetch market data for *{symbol}* (all venues failed)."

    prompt = prompt_for_pair(symbol, timeframe, snapshot, price)
    raw = call_gemini(prompt, GEMINI_MODEL_MANUAL)
    data = extract_json(raw)

    if not data:
        return "⚠️ Gemini JSON parse error. Raw snippet:\n\n" + raw[:2000]

    probs = data.get("probabilities", {})
    up = probs.get("upside", 0)
    down = probs.get("downside", 0)
    flat = probs.get("flat", 0)
    best = data.get("best_direction", "flat")
    view = data.get("overall_view", "")

    tp = data.get("trade_plan", {}) or {}
    direction = tp.get("direction", "none")
    rr_ratio = float(tp.get("rr_ratio", 0) or 0.0)

    # Enforce RR rule
    if rr_ratio < MIN_RR:
        direction = "none"

    tf_label = timeframe if timeframe else "Multi-timeframe"

    lines = []
    lines.append(f"📊 *{symbol}* — *{tf_label}*")
    lines.append(f"Price: `{price}`\n")
    lines.append("*Probabilities (next move):*")
    lines.append(f"⬆️ Upside: `{up}%`")
    lines.append(f"⬇️ Downside: `{down}%`")
    lines.append(f"➖ Flat: `{flat}%`")
    lines.append(f"🎯 Best direction: *{best.upper()}*\n")

    if view:
        lines.append(f"*View:* {view}\n")

    if direction != "none":
        lines.append("*🔥 Trade Setup (AI idea):*")
        lines.append(f"Direction: *{direction.upper()}*")
        lines.append(f"Entry: `{tp.get('entry', 0)}`")
        lines.append(f"SL (key level): `{tp.get('stop_loss', 0)}`")
        lines.append(f"TP1: `{tp.get('take_profit_1', 0)}`")
        lines.append(f"TP2: `{tp.get('take_profit_2', 0)}`")
        lines.append(f"RR: `{rr_ratio}` (≥ {MIN_RR})")
        lines.append(f"Leverage hint: `{tp.get('leverage_hint', '')}`")
        lines.append(f"Confidence: `{tp.get('confidence', 0)}%`")
        if tp.get("reasoning"):
            lines.append(f"Reason: {tp['reasoning']}")
    else:
        lines.append(
            f"🚫 No high-quality setup (probability < {MIN_PROB_MANUAL}% "
            f"or RR < {MIN_RR}). Better to wait."
        )

    lines.append("\n_Not financial advice. Manage your own risk._")
    return "\n".join(lines)

# ============================================================
# SCANNER ANALYSIS (FOR AUTO SIGNALS + AUTO-TRADE)
# ============================================================

def analyze_scan(symbol: str):
    snapshot, price = build_snapshot(symbol, SCAN_TIMEFRAMES)
    if price is None:
        return None

    prompt = prompt_for_scan(symbol, snapshot, price)
    raw = call_gemini(prompt, GEMINI_MODEL_SCANNER)
    data = extract_json(raw)
    if not data:
        return None

    probs = data.get("probabilities", {})
    up = float(probs.get("upside", 0) or 0.0)
    down = float(probs.get("downside", 0) or 0.0)
    flat = float(probs.get("flat", 0) or 0.0)
    best = data.get("best_direction", "flat")

    tp = data.get("trade_plan", {}) or {}
    direction = tp.get("direction", "none")
    rr_ratio = float(tp.get("rr_ratio", 0) or 0.0)

    if direction == "none" or best == "flat":
        return None

    prob = up if best == "upside" else down
    if prob < MIN_PROB_SCAN:
        return None
    if rr_ratio < MIN_RR:
        return None

    entry = float(tp.get("entry", 0) or 0.0)
    sl = float(tp.get("stop_loss", 0) or 0.0)
    tp1 = float(tp.get("take_profit_1", 0) or 0.0)
    tp2 = float(tp.get("take_profit_2", 0) or 0.0)
    conf = float(tp.get("confidence", prob) or prob)

    if entry <= 0 or sl <= 0 or tp1 <= 0:
        return None

    return {
        "symbol": symbol,
        "direction": direction,
        "probability": prob,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rr": rr_ratio,
        "confidence": conf,
        "flat": flat,
    }

# ============================================================
# BINGX AUTO-TRADING HELPERS
# ============================================================

def get_bingx_usdt_balance():
    if not bingx:
        return None
    try:
        info = bingx.get_perpetual_balance()
        # py-bingx usually returns {"data": [ {asset: "USDT", availableBalance: "..."} ]}
        if isinstance(info, dict) and "data" in info:
            for item in info["data"]:
                if str(item.get("asset", "")).upper() == "USDT":
                    return float(
                        item.get("availableBalance")
                        or item.get("available")
                        or item.get("balance")
                        or 0.0
                    )
    except Exception:
        return None
    return None


def maybe_auto_trade(sig: dict, context: CallbackContext):
    global auto_open_positions

    if not bingx or OWNER_CHAT_ID == 0:
        return

    # soft limit by our own tracking
    if len(auto_open_positions) >= AUTO_MAX_POSITIONS:
        context.bot.send_message(
            chat_id=OWNER_CHAT_ID,
            text=(
                f"⚠️ Auto-trade skipped for {sig['symbol']}: "
                f"max {AUTO_MAX_POSITIONS} auto positions reached."
            ),
        )
        return

    balance = get_bingx_usdt_balance()
    if balance is None or balance <= 0:
        context.bot.send_message(
            chat_id=OWNER_CHAT_ID,
            text="⚠️ Auto-trade skipped: BingX USDT balance unavailable or zero.",
        )
        return

    entry = sig["entry"]
    if entry <= 0:
        context.bot.send_message(
            chat_id=OWNER_CHAT_ID,
            text=f"⚠️ Auto-trade skipped for {sig['symbol']}: invalid entry.",
        )
        return

    total_notional = balance * AUTO_LEVERAGE
    per_slot_notional = total_notional / AUTO_MAX_POSITIONS
    qty = per_slot_notional / entry
    if qty <= 0:
        context.bot.send_message(
            chat_id=OWNER_CHAT_ID,
            text=f"⚠️ Auto-trade skipped for {sig['symbol']}: qty <= 0.",
        )
        return

    bingx_symbol = binance_to_bingx_symbol(sig["symbol"])
    side = "LONG" if sig["direction"].lower() == "long" else "SHORT"
    qty_str = f"{qty:.8f}"

    try:
        order = bingx.open_market_order(
            bingx_symbol,
            side,
            qty_str,
            tp=str(sig["tp1"]),
            sl=str(sig["sl"]),
        )
        auto_open_positions.add(sig["symbol"])

        context.bot.send_message(
            chat_id=OWNER_CHAT_ID,
            parse_mode="Markdown",
            text=(
                "✅ *Auto-trade executed on BingX*\n"
                f"Pair: *{bingx_symbol}* ({sig['symbol']})\n"
                f"Side: *{side}*\n"
                f"Entry (signal): `{sig['entry']}`\n"
                f"SL: `{sig['sl']}`\n"
                f"TP1: `{sig['tp1']}`\n"
                f"RR: `{sig['rr']}`\n"
                f"Used balance: ~`100%` at `{AUTO_LEVERAGE}x`, "
                f"split across `{AUTO_MAX_POSITIONS}` slots.\n\n"
                "_Always verify orders and manage risk._"
            ),
        )
    except Exception as e:
        context.bot.send_message(
            chat_id=OWNER_CHAT_ID,
            text=f"❌ Auto-trade failed for {bingx_symbol}: `{e}`",
            parse_mode="Markdown",
        )

# ============================================================
# TELEGRAM HANDLERS
# ============================================================

def start(update: Update, context: CallbackContext):
    global SCAN_ENABLED
    SCAN_ENABLED = True

    text = (
        "🤖 *Gemini Auto Futures Bot + BingX Auto-Trade*\n\n"
        "Scanner is now: *ON* ✅\n\n"
        "Commands:\n"
        "• `/suiusdt` → Multi-timeframe AI analysis\n"
        "• `/suiusdt 4h` → Single timeframe analysis\n"
        "• `/stop` → Stop auto scanner (manual analysis still works)\n"
        "• `/start` → Turn auto scanner ON again\n\n"
        "Scanner filters:\n"
        f"• 24h futures volume ≥ `{MIN_VOLUME:,}` USDT\n"
        f"• Probability(upside/downside) ≥ `{MIN_PROB_SCAN}%`\n"
        f"• RR ≥ `1:{MIN_RR}`\n\n"
        "Auto-trade rules (BingX USDT-M Perp):\n"
        f"• Uses ~`100%` of available USDT at `{AUTO_LEVERAGE}x`\n"
        f"• Max `{AUTO_MAX_POSITIONS}` open auto positions at once\n"
        "• For 2 positions, notional is split equally.\n\n"
        "_HIGH RISK: This is not financial advice. Use at your own risk._"
    )
    update.message.reply_markdown(text)


def stop(update: Update, context: CallbackContext):
    global SCAN_ENABLED
    SCAN_ENABLED = False
    update.message.reply_text(
        "⏹ Auto scanner is now *OFF*.\n\n"
        "Manual analysis like `/btcusdt` or `/ethusdt 1h` still works.\n"
        "Send `/start` to turn the scanner ON again."
    )


def handle_pair(update: Update, context: CallbackContext):
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    parts = text.split()

    symbol = parts[0].lstrip("/").upper()
    timeframe = parts[1].lower() if len(parts) > 1 else None

    update.message.reply_text(
        f"⏳ Analysing {symbol}"
        + (f" on {timeframe}" if timeframe else "")
        + " with Gemini..."
    )

    try:
        result = analyze_command(symbol, timeframe)
    except Exception as e:
        result = f"❌ Error while analysing {symbol}: {e}"

    update.message.reply_markdown(result)

# ============================================================
# SCANNER JOB
# ============================================================

def scanner_job(context: CallbackContext):
    if OWNER_CHAT_ID == 0:
        return
    if not SCAN_ENABLED:
        return

    try:
        symbols = get_top_symbols()
    except Exception:
        return

    now = datetime.now(timezone.utc)

    for sym in symbols:
        try:
            sig = analyze_scan(sym)
        except Exception:
            continue

        if not sig:
            continue

        key = (sym, sig["direction"])
        last = last_signal_time.get(key)
        if last and (now - last).total_seconds() < 1800:  # 30 min cooldown per direction
            continue

        last_signal_time[key] = now

        msg = (
            "🚨 *AI Scanner Signal*\n"
            f"Pair: *{sym}*\n"
            f"Direction: *{sig['direction'].upper()}*\n"
            f"Probability: `{sig['probability']}%` (≥ {MIN_PROB_SCAN}%)\n"
            f"RR: `{sig['rr']}` (≥ {MIN_RR})\n"
            f"Entry: `{sig['entry']}`\n"
            f"SL (key level): `{sig['sl']}`\n"
            f"TP1: `{sig['tp1']}`\n"
            f"TP2: `{sig['tp2']}`\n"
            f"Confidence: `{sig['confidence']}%`\n"
            f"Flat probability: `{sig['flat']}%`\n\n"
            "_Signal via Gemini AI. Use your own risk management._"
        )

        context.bot.send_message(
            chat_id=OWNER_CHAT_ID,
            text=msg,
            parse_mode="Markdown",
        )

        # try to auto-trade this signal
        maybe_auto_trade(sig, context)

# ============================================================
# MAIN
# ============================================================

def main():
    updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", start))
    dp.add_handler(CommandHandler("stop", stop))

    # Any /COIN or /COIN timeframe command
    dp.add_handler(MessageHandler(Filters.command, handle_pair))

    jq = updater.job_queue
    jq.run_repeating(scanner_job, interval=SCAN_INTERVAL_SECONDS, first=30)

    print("✅ Bot running with polling + scanner + BingX auto-trade...")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
