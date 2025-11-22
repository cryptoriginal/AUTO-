import os
import json
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

# BingX wrapper
from bingx.api import BingxAPI  # from py-bingx


# ===============================
# ENVIRONMENT VARIABLES
# ===============================

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
GEMINI_MODEL_SCANNER = "gemini-2.5-flash"  # you can change to cheaper model later

# ===============================
# STRATEGY SETTINGS
# ===============================

SCAN_ENABLED = True  # controlled by /start and /stop

SCAN_INTERVAL_SECONDS = 300        # 5 minutes
MIN_VOLUME = 50_000_000            # 24h quote volume filter (>= 50M)
MAX_SCAN_SYMBOLS = 25

# probability thresholds
MIN_PROB_MANUAL = 75               # for manual /pair analysis
MIN_PROB_SCAN = 85                 # for auto-scan + auto-trade

# RR constraints
MIN_RR = 2.1                       # minimum RR 1:2.1

DEFAULT_TIMEFRAMES = [
    "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"
]
SCAN_TIMEFRAMES = ["15m", "1h", "4h"]

BINANCE_FAPI = "https://fapi.binance.com"

# auto-trading controls
AUTO_MAX_POSITIONS = 2          # max auto positions at once
AUTO_LEVERAGE = 3.0             # use 3x notional vs equity

last_signal_time = {}           # (symbol, direction) -> datetime
auto_open_positions = set()     # track symbols where bot opened a position

# Gemini client
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# BingX client (can be None if no keys)
bingx = None
if BINGX_API_KEY and BINGX_API_SECRET:
    bingx = BingxAPI(BINGX_API_KEY, BINGX_API_SECRET, timestamp="local")


# ===============================
# BINANCE HELPERS
# ===============================

def get_klines(symbol, interval, limit=120):
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def get_top_symbols():
    url = f"{BINANCE_FAPI}/fapi/v1/ticker/24hr"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()

    pairs = [
        s for s in data
        if s.get("symbol", "").endswith("USDT")
        and float(s.get("quoteVolume", 0.0)) >= MIN_VOLUME
    ]
    pairs.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return [p["symbol"] for p in pairs[:MAX_SCAN_SYMBOLS]]


def build_snapshot(symbol, timeframes):
    snapshot = {}
    current_price = None

    for tf in timeframes:
        kl = get_klines(symbol, tf, 100)
        candles = []
        for c in kl:
            candles.append(
                {
                    "open_time": c[0],
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


# ===============================
# GEMINI PROMPTS
# ===============================

def prompt_for_pair(symbol, timeframe, snapshot, price):
    return f"""
You are an elite crypto futures analyst and trader.

Symbol: {symbol}
Current price: {price}
Timeframe focus: {timeframe if timeframe else "Multi-timeframe"}

OHLCV JSON:
{json.dumps(snapshot)[:80000]}

MAIN ANALYSIS RULES:
- Focus on PRICE ACTION + MARKET STRUCTURE:
  - Trend (up / down / range) on each TF.
  - Major support/resistance & supply-demand zones.
  - Trendlines / channels, breakouts & retests.
  - Reversal candles (hammer, shooting star, engulfing, pin bars) at key levels.
  - Patterns (triangles, flags, head & shoulders, double tops/bottoms, etc.).
- Use a FIXED RANGE VOLUME PROFILE mental model:
  - SL must sit beyond a clear invalidation level (HVN/LVN or swing high/low).
- Use EMAs (20/50/200), RSI, MACD, volume only as confirmation.

TAKE PROFIT / RISK:
- Risk:Reward rr_ratio must be >= {MIN_RR} (1:{MIN_RR}).
- TP1 must be realistic and consistent with the upside/downside probability.
- TP2 can be more ambitious but still logically reachable.

TASK:
1. Estimate probabilities (0-100):
   - upside
   - downside
   - flat

2. Decide best_direction.

3. ONLY IF:
   - best_direction is upside or downside, AND
   - its probability >= {MIN_PROB_MANUAL}, AND
   - a clean setup with rr_ratio >= {MIN_RR} exists,
   THEN produce a trade plan with direction, entry, stop_loss, take_profit_1, take_profit_2,
   rr_ratio, leverage_hint, confidence and reasoning.

4. Otherwise, direction="none" and rr_ratio=0.

Return ONLY JSON:

{{
  "symbol": "{symbol}",
  "probabilities": {{
    "upside": 0,
    "downside": 0,
    "flat": 0
  }},
  "best_direction": "upside | downside | flat",
  "overall_view": "text",
  "trade_plan": {{
    "direction": "long | short | none",
    "entry": 0,
    "stop_loss": 0,
    "take_profit_1": 0,
    "take_profit_2": 0,
    "rr_ratio": 0,
    "leverage_hint": "",
    "confidence": 0,
    "reasoning": "text"
  }}
}}
"""


def prompt_for_scan(symbol, snapshot, price):
    return f"""
Quick scanner for crypto futures pair:

Symbol: {symbol}
Price: {price}

OHLCV JSON:
{json.dumps(snapshot)[:60000]}

RULES:
- Use price action, trend, key levels, breakouts, reversals.
- Use a fixed range volume profile mental model for SL (beyond invalidation level).
- You must strictly enforce:
  - best_direction is upside or downside (not flat),
  - probability of that move >= {MIN_PROB_SCAN}%, and
  - rr_ratio >= {MIN_RR} (1:{MIN_RR} or better).

- TP1 must be realistic and match the move probability.
- TP2 can be more ambitious but still logical.

Return ONLY JSON:

{{
 "symbol": "{symbol}",
 "probabilities": {{
   "upside": 0,
   "downside": 0,
   "flat": 0
 }},
 "best_direction": "",
 "trade_plan": {{
   "direction": "",
   "entry": 0,
   "stop_loss": 0,
   "take_profit_1": 0,
   "take_profit_2": 0,
   "rr_ratio": 0,
   "confidence": 0
 }}
}}
"""


# ===============================
# GEMINI CALL + JSON PARSE
# ===============================

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


# ===============================
# ANALYSIS FUNCTIONS (MANUAL)
# ===============================

def analyze_command(symbol: str, timeframe: str | None):
    tfs = [timeframe] if timeframe else DEFAULT_TIMEFRAMES
    snapshot, price = build_snapshot(symbol, tfs)

    if price is None:
        return f"❌ Could not fetch market data for *{symbol}*."

    prompt = prompt_for_pair(symbol, timeframe, snapshot, price)
    raw = call_gemini(prompt, GEMINI_MODEL_MANUAL)
    data = extract_json(raw)

    if not data:
        return "⚠️ Gemini JSON parse error. Raw:\n\n" + raw[:2500]

    probs = data.get("probabilities", {})
    up = probs.get("upside", 0)
    down = probs.get("downside", 0)
    flat = probs.get("flat", 0)
    best = data.get("best_direction", "flat")
    view = data.get("overall_view", "")

    tp = data.get("trade_plan", {}) or {}
    direction = tp.get("direction", "none")
    rr_ratio = float(tp.get("rr_ratio", 0) or 0.0)

    # Enforce RR threshold for manual proposals too
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
        lines.append("*🔥 Trade Setup (AI):*")
        lines.append(f"Direction: *{direction.upper()}*")
        lines.append(f"Entry: `{tp.get('entry', 0)}`")
        lines.append(f"SL (key level): `{tp.get('stop_loss', 0)}`")
        lines.append(f"TP1: `{tp.get('take_profit_1', 0)}`")
        lines.append(f"TP2: `{tp.get('take_profit_2', 0)}`")
        lines.append(f"RR: `{rr_ratio}` (≥ {MIN_RR})")
        lines.append(f"Leverage: `{tp.get('leverage_hint', '')}`")
        lines.append(f"Confidence: `{tp.get('confidence', 0)}%`")
        if tp.get("reasoning"):
            lines.append(f"Reason: {tp['reasoning']}")
    else:
        lines.append(
            f"🚫 No strong setup (probability < {MIN_PROB_MANUAL}% "
            f"or RR < {MIN_RR})."
        )

    lines.append("\n_Not financial advice. Manage your own risk._")
    return "\n".join(lines)


# ===============================
# ANALYSIS FOR SCANNER
# ===============================

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
    flat = float(probs.get("flat", 0) or 0.0)  # not directly used
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

    return {
        "symbol": symbol,
        "direction": direction,
        "probability": prob,
        "entry": float(tp.get("entry", 0) or 0.0),
        "sl": float(tp.get("stop_loss", 0) or 0.0),
        "tp1": float(tp.get("take_profit_1", 0) or 0.0),
        "tp2": float(tp.get("take_profit_2", 0) or 0.0),
        "rr": rr_ratio,
        "confidence": float(tp.get("confidence", prob) or prob),
    }


# ===============================
# BINGX HELPERS (AUTO-TRADE)
# ===============================

def binance_to_bingx_symbol(symbol: str) -> str:
    # BTCUSDT -> BTC-USDT, SUIUSDT -> SUI-USDT, etc.
    if symbol.endswith("USDT"):
        return symbol[:-4] + "-USDT"
    return symbol


def get_bingx_usdt_balance():
    """
    Try to read available USDT balance from BingX perpetual account.

    NOTE: py-bingx returns a dict/list depending on version.
    If this fails, check logs and adjust parsing.
    """
    if not bingx:
        return None

    info = bingx.get_perpetual_balance()
    usdt_balance = None

    def extract_from_obj(obj):
        for key in ["availableBalance", "available", "balance"]:
            if key in obj:
                try:
                    return float(obj[key])
                except Exception:
                    continue
        return None

    try:
        # common patterns
        if isinstance(info, dict):
            if "data" in info and isinstance(info["data"], list):
                for item in info["data"]:
                    if str(item.get("asset", "")).upper() == "USDT":
                        bal = extract_from_obj(item)
                        if bal is not None:
                            return bal
            bal = extract_from_obj(info)
            if bal is not None:
                return bal

        if isinstance(info, list):
            for item in info:
                if str(item.get("asset", "")).upper() == "USDT":
                    bal = extract_from_obj(item)
                    if bal is not None:
                        return bal
    except Exception:
        return None

    return usdt_balance


def maybe_auto_trade(sig: dict, context: CallbackContext):
    """
    Auto-trade on BingX if:
    - bingx client available
    - less than AUTO_MAX_POSITIONS currently open (tracked by bot)
    Uses 100% balance * AUTO_LEVERAGE, split equally across AUTO_MAX_POSITIONS slots.
    """
    global auto_open_positions

    if not bingx or OWNER_CHAT_ID == 0:
        return

    # soft limit by bot tracking (doesn't know about manual closes)
    if len(auto_open_positions) >= AUTO_MAX_POSITIONS:
        context.bot.send_message(
            chat_id=OWNER_CHAT_ID,
            text="⚠️ Auto-trade skipped for "
                 f"{sig['symbol']} — max auto positions reached.",
        )
        return

    balance = get_bingx_usdt_balance()
    if balance is None or balance <= 0:
        context.bot.send_message(
            chat_id=OWNER_CHAT_ID,
            text="⚠️ Auto-trade skipped: could not read BingX USDT balance "
                 "or balance is zero.",
        )
        return

    entry = sig["entry"]
    if entry <= 0:
        context.bot.send_message(
            chat_id=OWNER_CHAT_ID,
            text=f"⚠️ Auto-trade skipped for {sig['symbol']}: invalid entry.",
        )
        return

    # total notional = equity * leverage; split into equal slots for 2 positions
    total_notional = balance * AUTO_LEVERAGE
    per_position_notional = total_notional / AUTO_MAX_POSITIONS
    qty = per_position_notional / entry

    if qty <= 0:
        context.bot.send_message(
            chat_id=OWNER_CHAT_ID,
            text=f"⚠️ Auto-trade skipped for {sig['symbol']}: qty <= 0.",
        )
        return

    bingx_symbol = binance_to_bingx_symbol(sig["symbol"])
    side = "LONG" if sig["direction"].lower() == "long" else "SHORT"

    qty_str = f"{qty:.8f}"  # you can fine-tune precision per pair
    tp_price = sig["tp1"]
    sl_price = sig["sl"]

    try:
        order_data = bingx.open_market_order(
            bingx_symbol,
            side,
            qty_str,
            tp=str(tp_price),
            sl=str(sl_price),
        )
        auto_open_positions.add(sig["symbol"])

        context.bot.send_message(
            chat_id=OWNER_CHAT_ID,
            text=(
                "✅ *Auto-trade executed on BingX*\n"
                f"Pair: *{bingx_symbol}* ({sig['symbol']})\n"
                f"Side: *{side}*\n"
                f"Entry (signal): `{entry}`\n"
                f"SL: `{sl_price}`\n"
                f"TP1 (linked to order): `{tp_price}`\n"
                f"Risk:Reward: `{sig['rr']}`\n"
                f"Used balance: ~`100%` at `{AUTO_LEVERAGE}x` "
                f"(split for {AUTO_MAX_POSITIONS} slots)\n\n"
                "_Check BingX to confirm exact fills and fees._"
            ),
            parse_mode="Markdown",
        )

    except Exception as e:
        context.bot.send_message(
            chat_id=OWNER_CHAT_ID,
            text=f"❌ Auto-trade failed for {bingx_symbol}: `{e}`",
            parse_mode="Markdown",
        )


# ===============================
# TELEGRAM HANDLERS
# ===============================

def start(update: Update, context: CallbackContext):
    global SCAN_ENABLED
    SCAN_ENABLED = True

    text = (
        "🤖 *Gemini 2.5 Futures Bot + BingX Auto-Trade*\n\n"
        "Scanner is now: *ON* ✅\n\n"
        "Commands:\n"
        "• `/coin` → Multi-timeframe AI analysis (e.g. `/suiusdt`)\n"
        "• `/coin timeframe` → Single TF analysis (e.g. `/suiusdt 4h`)\n"
        "• `/stop` → Stop auto scanner (manual analysis still works)\n"
        "• `/start` → Turn auto scanner ON again\n\n"
        "Scanner rules:\n"
        f"• Scans Binance USDT-M futures with 24h vol ≥ {MIN_VOLUME:,} USDT\n"
        f"• Signals only if upside/downside prob ≥ {MIN_PROB_SCAN}%\n"
        f"• Trade RR must be ≥ 1:{MIN_RR}\n\n"
        "Auto-Trade (BingX):\n"
        f"• Uses *100%* of USDT balance at *{AUTO_LEVERAGE}x*\n"
        f"• Max *{AUTO_MAX_POSITIONS}* open auto positions at a time\n"
        "• For 2 positions, notional is split equally.\n\n"
        "_HIGH RISK: Always monitor your account. This is NOT financial advice._"
    )
    update.message.reply_markdown(text)


def stop(update: Update, context: CallbackContext):
    global SCAN_ENABLED
    SCAN_ENABLED = False
    update.message.reply_text(
        "⏹ Auto scanner is now *OFF*.\n\n"
        "You can still use manual analysis like `/btcusdt` or `/ethusdt 4h`.\n"
        "Send `/start` to turn auto scanner ON again."
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
        + " using Gemini 2.5..."
    )

    try:
        result = analyze_command(symbol, timeframe)
    except Exception as e:
        result = f"❌ Error while analysing {symbol}: {e}"

    update.message.reply_markdown(result)


# ===============================
# SCANNER JOB
# ===============================

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

        key = (sig["symbol"], sig["direction"])
        last = last_signal_time.get(key)
        if last and (now - last).total_seconds() < 1800:  # 30 min cooldown
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
            f"Confidence: `{sig['confidence']}%`\n\n"
            "_Signal is generated via Gemini AI. Use your own risk management._"
        )

        context.bot.send_message(
            chat_id=OWNER_CHAT_ID,
            text=msg,
            parse_mode="Markdown",
        )

        # try to auto-trade this signal on BingX
        maybe_auto_trade(sig, context)


# ===============================
# MAIN
# ===============================

def main():
    updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", start))
    dp.add_handler(CommandHandler("stop", stop))

    # any /COIN or /COIN timeframe
    dp.add_handler(MessageHandler(Filters.command, handle_pair))

    jq = updater.job_queue
    jq.run_repeating(scanner_job, interval=SCAN_INTERVAL_SECONDS, first=30)

    print("✅ Bot running with polling + scanner + BingX auto-trade (PTB v13.15)...")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
