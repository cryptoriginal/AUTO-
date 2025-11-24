# app.py
import os
import time
import json
import logging
from datetime import datetime, timezone

import requests
from google import genai
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ============================================================
# CONFIG / ENV
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "0") or "0")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN env var is required")

# Gemini client uses GEMINI_API_KEY env by default
gemini_client = genai.Client()

MEXC_FUTURES_BASE = "https://contract.mexc.com"
SCAN_INTERVAL_SEC = 5 * 60      # every 5 minutes
COOLDOWN_SEC = 10 * 60          # 10-minute cooldown after sending signals
MAX_SCAN_PAIRS = 15             # limit number of pairs per scan to control cost/time

# Map user timeframe strings to MEXC interval and label
TIMEFRAME_MAP = {
    "5m": ("Min5", "5m"),
    "15m": ("Min15", "15m"),
    "1h": ("Min60", "1h"),
    "2h": ("Min60", "2h (approx, using 1h data)"),
    "4h": ("Hour4", "4h"),
    "6h": ("Hour4", "6h (approx, using 4h data)"),
    "12h": ("Hour8", "12h (approx, using 8h data)"),
    "1d": ("Day1", "1D"),
    "d": ("Day1", "1D"),
    "daily": ("Day1", "1D"),
    "1w": ("Week1", "1W"),
    "w": ("Week1", "1W"),
    "weekly": ("Week1", "1W"),
}

# Default multi-timeframe set for manual analysis
MULTI_TF_DEFAULTS = [
    ("Min5", "5m"),
    ("Min15", "15m"),
    ("Min60", "1h"),
    ("Hour4", "4h"),
    ("Day1", "1D"),
    ("Week1", "1W"),
]

# In-memory storage of autoscan jobs per chat
AUTOSCAN_JOBS = {}

# Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ============================================================
# MEXC HELPERS
# ============================================================

def fetch_mexc_klines(symbol: str, interval: str, limit: int = 150):
    """
    Fetch OHLCV candles from MEXC futures for given symbol/interval.
    Uses start/end timestamps to limit to ~`limit` candles.
    Returns list of dicts: {"time", "open", "high", "low", "close", "volume"}
    """
    # Approx seconds per bar for supported intervals
    seconds_per_bar = {
        "Min1": 60,
        "Min5": 5 * 60,
        "Min15": 15 * 60,
        "Min30": 30 * 60,
        "Min60": 60 * 60,
        "Hour4": 4 * 60 * 60,
        "Hour8": 8 * 60 * 60,
        "Day1": 24 * 60 * 60,
        "Week1": 7 * 24 * 60 * 60,
        "Month1": 30 * 24 * 60 * 60,
    }
    now = int(time.time())
    span = seconds_per_bar.get(interval, 60 * 60) * limit
    start_ts = now - span
    params = {
        "interval": interval,
        "start": start_ts,
        "end": now,
    }

    url = f"{MEXC_FUTURES_BASE}/api/v1/contract/kline/{symbol}"
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            logger.warning(f"MEXC kline error for {symbol}: {data}")
            return []

        d = data.get("data", {})
        times = d.get("time") or []
        opens = d.get("open") or []
        highs = d.get("high") or []
        lows = d.get("low") or []
        closes = d.get("close") or []
        vols = d.get("vol") or []

        candles = []
        for i in range(min(len(times), len(opens), len(highs), len(lows), len(closes), len(vols))):
            candles.append(
                {
                    "time": int(times[i]),
                    "open": float(opens[i]),
                    "high": float(highs[i]),
                    "low": float(lows[i]),
                    "close": float(closes[i]),
                    "volume": float(vols[i]),
                }
            )
        return candles
    except Exception as e:
        logger.exception(f"Error fetching MEXC kline for {symbol}: {e}")
        return []


def fetch_high_volume_symbols(min_turnover_usdt: float = 50_000_000.0, max_pairs: int = MAX_SCAN_PAIRS):
    """
    Fetch futures tickers from MEXC and filter for:
      - USDT-margined contracts (symbol ends with _USDT)
      - 24h turnover (amount24) >= min_turnover_usdt
    Returns list of symbol strings like "BTC_USDT".
    """
    url = f"{MEXC_FUTURES_BASE}/api/v1/contract/ticker"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("data")

        # API may return object or list; normalize to list
        if isinstance(raw, dict):
            items = [raw]
        elif isinstance(raw, list):
            items = raw
        else:
            items = []

        filtered = []
        for item in items:
            symbol = item.get("symbol")
            amount24 = float(item.get("amount24", 0.0))
            if not symbol or not symbol.endswith("_USDT"):
                continue
            if amount24 >= min_turnover_usdt:
                filtered.append((symbol, amount24))

        # Sort by turnover desc and keep top N
        filtered.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, _ in filtered[:max_pairs]]
        return top_symbols
    except Exception as e:
        logger.exception(f"Error fetching MEXC tickers: {e}")
        return []


def symbol_from_command(cmd: str) -> str:
    """
    Convert Telegram command like 'suiusdt' into MEXC symbol 'SUI_USDT'.
    """
    cmd = cmd.strip().lstrip("/").upper()
    if cmd.endswith("USDT"):
        base = cmd[:-4]
        return f"{base}_USDT"
    # fallback: assume already "SUI_USDT" style
    if "_" in cmd:
        return cmd
    return f"{cmd}_USDT"


# ============================================================
# GEMINI PROMPTS / HELPERS
# ============================================================

def format_ohlcv_for_prompt(candles, max_rows=150):
    """
    Format OHLCV into a compact CSV-style block for LLM.
    Oldest to newest.
    """
    candles = candles[-max_rows:]
    lines = ["time,open,high,low,close,volume"]
    for c in candles:
        ts = int(c["time"])
        # convert key to readable UTC time (optional)
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        lines.append(
            f"{dt},{c['open']:.6f},{c['high']:.6f},{c['low']:.6f},{c['close']:.6f},{c['volume']:.2f}"
        )
    return "\n".join(lines)


def build_manual_analysis_prompt(symbol: str, tf_blocks: dict, requested_tf_label: str | None):
    """
    Build the prompt for manual /coin analysis.
    tf_blocks: { "5m": "csv string", "15m": "csv string", ... }
    """
    scope_text = (
        f"Focus ONLY on timeframe: {requested_tf_label}.\n"
        "Treat it as the main decision frame. Ignore other timeframes."
        if requested_tf_label
        else "Use MULTI-TIMEFRAME analysis: weekly & daily for bias, 4h/1h for structure, 15m/5m for timing."
    )

    tf_texts = []
    for label, csv_block in tf_blocks.items():
        tf_texts.append(f"=== TIMEFRAME {label} ===\n{csv_block}")

    prompt = f"""
You are a world-class crypto futures trader and risk manager (top 1% on earth).

Exchange: MEXC Futures
Contract: {symbol}

Task:
- Analyze the provided OHLCV data.
- Estimate the probability (in %) of three scenarios in the NEXT phase of the market:
  1) Upside move (sustained bullish move)
  2) Downside move (sustained bearish move)
  3) Flat / choppy / range-bound conditions where trade should be avoided

{scope_text}

When reasoning:
- Consider trend, market structure (HH/HL, LH/LL), key support/resistance, liquidity zones.
- Consider candlestick patterns (hammer, doji, engulfing, pin bars), wicks, rejection candles.
- Consider classic chart patterns (flags, wedges, ranges).
- Approximate leading indicators (RSI, MACD, moving averages) mentally from the OHLCV.
- Use conservative risk management: only suggest trades when edge is clear.

Trade rules:
- Let "highest_prob" be the scenario with the highest probability.
- If highest_prob < 75, DO NOT provide a trade plan (trade_plan = null).
- If highest_prob >= 75:
    - If dominant scenario is upside: suggest a LONG.
    - If dominant scenario is downside: suggest a SHORT.
    - Entry should be near a logical level (e.g. retest of broken structure, key S/R, VWAP-like levels).
    - Stop loss must be beyond a meaningful swing high/low to reduce stop-hunt risk.
    - Provide 1-2 take profit targets with minimum risk:reward of 1:1.9 or better.

Output format:
Return STRICTLY valid JSON, no extra text, no markdown, no commentary.

JSON SCHEMA:
{{
  "upside_prob": int,   // 0-100
  "downside_prob": int, // 0-100
  "flat_prob": int,     // 0-100
  "dominant_scenario": "upside" | "downside" | "flat",
  "trade_plan": null | {{
     "direction": "long" | "short",
     "entry": float,
     "stop_loss": float,
     "take_profits": [float, ...],
     "min_rr": float
  }},
  "summary": "very short explanation, max 3 sentences"
}}

Make sure upside_prob + downside_prob + flat_prob is close to 100.
Return ONLY this JSON object.
"""
    full_contents = [prompt, "\n\n".join(tf_texts)]
    return full_contents


def build_autoscan_prompt(symbol: str, tf_label: str, csv_block: str):
    """
    Build prompt for auto-scan signals.
    Focus: VWAP + fixed range volume profile between last swing high/low + candles.
    Threshold for trade_plan: 85%.
    """
    prompt = f"""
You are a crypto futures scalping expert, focused on VWAP and fixed-range volume profile.

Exchange: MEXC Futures
Contract: {symbol}
Timeframe: {tf_label}

Data:
- OHLCV data is provided in CSV form: time,open,high,low,close,volume.
- Use it to approximate:
    - Session VWAP and intraday VWAP behaviour.
    - Fixed-range volume profile between most recent major swing high and swing low (approximate using per-candle volume and price).
    - Identification of high-volume nodes / low-volume areas.
- Combine this with candlestick patterns and basic structure to make decisions.

Task:
- Estimate the probability (in %) of:
  1) Upside move (trend move up)
  2) Downside move (trend move down)
  3) Flat / choppy / noisy zone

- Pay SPECIAL attention to:
  - Price interaction with VWAP (bounces, rejections, reclaim/reject patterns).
  - Price interacting with high-volume nodes or low-volume areas within the recent swing range.
  - Candlestick confirmation at those levels (rejection wicks, engulfing, hammers, etc).

Trade rules:
- Let "highest_prob" be the scenario with the highest probability.
- If highest_prob < 85, set trade_plan = null (NO trade).
- If highest_prob >= 85:
    - If dominant scenario is upside: LONG.
    - If dominant scenario is downside: SHORT.
    - Choose entry at a logical level that aligns with VWAP + volume profile + structure.
    - Stop loss must be beyond a meaningful swing high/low.
    - Provide 1-2 take profit levels with minimum risk:reward of 1:1.9 or better.

Output format:
Return STRICTLY valid JSON, no extra text.

JSON SCHEMA:
{{
  "upside_prob": int,
  "downside_prob": int,
  "flat_prob": int,
  "dominant_scenario": "upside" | "downside" | "flat",
  "trade_plan": null | {{
     "direction": "long" | "short",
     "entry": float,
     "stop_loss": float,
     "take_profits": [float, ...],
     "min_rr": float
  }}
}}

Return ONLY this JSON object.
"""

    return [prompt, f"=== TIMEFRAME {tf_label} ===\n{csv_block}"]


def call_gemini_for_json(contents):
    """
    Call Gemini 2.5 Flash and parse JSON from response.text.
    Returns dict or None.
    """
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )
        raw = response.text
        # Sometimes models wrap JSON in ```json ```; strip if needed.
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            # Might be like json\n{...}
            if "\n" in raw:
                raw = raw.split("\n", 1)[1]
        data = json.loads(raw)
        return data
    except Exception as e:
        logger.exception(f"Error calling Gemini or parsing JSON: {e}")
        return None


# ============================================================
# TELEGRAM HANDLERS
# ============================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /start:
      - Send welcome message
      - Start autoscan job for this chat (if not already)
    """
    chat_id = update.effective_chat.id

    if chat_id in AUTOSCAN_JOBS:
        await update.message.reply_text(
            "✅ Bot is already running auto-scan.\n\n"
            "Use /stop to stop auto-signals.\n"
            "You can still request manual analysis: e.g. /btcusdt 1h"
        )
        return

    job = context.job_queue.run_repeating(
        autoscan_job,
        interval=SCAN_INTERVAL_SEC,
        first=5,  # start after 5 seconds
        data={"chat_id": chat_id, "last_signal_time": 0.0},
        name=f"autoscan_{chat_id}",
    )
    AUTOSCAN_JOBS[chat_id] = job

    await update.message.reply_text(
        "🚀 Auto-scan started.\n\n"
        "- Every 5 minutes I scan MEXC USDT futures with ≥ 50M 24h volume.\n"
        "- I only send signals when probability ≥ 85% and RR ≥ 1:1.9.\n"
        "- I focus on VWAP + fixed-range volume profile + candle confirmation.\n"
        "- There is a ~10 minute cooldown after sending signals.\n\n"
        "You can still request manual analysis anytime, e.g.:\n"
        "`/suiusdt 4h` or `/btcusdt` (multi-timeframe)\n\n"
        "⚠️ Futures trading is risky. Use position sizing and your own judgment.",
        parse_mode="Markdown",
    )


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /stop: stop autoscan for this chat.
    """
    chat_id = update.effective_chat.id
    job = AUTOSCAN_JOBS.pop(chat_id, None)
    if job:
        job.schedule_removal()
        await update.message.reply_text(
            "🛑 Auto-scan stopped.\n\n"
            "You can still ask for manual analysis, e.g. `/ethusdt 1h`.",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text("Auto-scan is not running for this chat.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📈 Gemini + MEXC Futures Analysis Bot\n\n"
        "**Manual analysis:**\n"
        "- `/btcusdt` → multi-timeframe analysis (5m → 1W)\n"
        "- `/suiusdt 4h` → analysis only on 4h\n\n"
        "I give probabilities of Upside / Downside / Flat.\n"
        "If the best scenario ≥ 75% I also suggest entry / SL / TP.\n\n"
        "**Auto-scan:**\n"
        "- `/start` → start auto-scan (5m, VWAP + volume profile focus)\n"
        "- `/stop` → stop auto-scan\n\n"
        "⚠️ Signals are NOT guaranteed. Use them as decision support only.",
        parse_mode="Markdown",
    )


async def autoscan_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Background job:
      - Runs every 5 minutes
      - Scans high-volume pairs
      - Calls Gemini for each candidate
      - Sends only strong signals (prob ≥ 85%, RR ≥ 1.9)
      - Enforces 10-min cooldown after sending signals
    """
    job_data = context.job.data or {}
    chat_id = job_data.get("chat_id")
    last_signal_time = float(job_data.get("last_signal_time", 0.0))
    now = time.time()

    # Cooldown check
    if last_signal_time and (now - last_signal_time) < COOLDOWN_SEC:
        logger.info(f"Autoscan cooldown active for chat {chat_id}")
        return

    symbols = fetch_high_volume_symbols()
    if not symbols:
        logger.info("No high-volume symbols found this scan.")
        return

    logger.info(f"Autoscan symbols: {symbols}")

    messages = []

    for symbol in symbols:
        candles = fetch_mexc_klines(symbol, "Min5", limit=200)
        if len(candles) < 50:
            continue

        csv = format_ohlcv_for_prompt(candles, max_rows=150)
        contents = build_autoscan_prompt(symbol, "5m", csv)
        data = call_gemini_for_json(contents)
        if not data:
            continue

        dominant = data.get("dominant_scenario")
        upside_prob = int(data.get("upside_prob", 0))
        downside_prob = int(data.get("downside_prob", 0))
        flat_prob = int(data.get("flat_prob", 0))
        trade_plan = data.get("trade_plan")

        highest = max(upside_prob, downside_prob, flat_prob)
        if highest < 85 or not trade_plan:
            continue

        direction = trade_plan.get("direction")
        entry = trade_plan.get("entry")
        sl = trade_plan.get("stop_loss")
        tps = trade_plan.get("take_profits") or []
        min_rr = trade_plan.get("min_rr", 0.0)

        if not direction or entry is None or sl is None or not tps:
            continue

        tp_str = ", ".join(f"{tp:.6f}" for tp in tps)
        msg = (
            f"📡 [AUTO] {symbol}\n"
            f"Direction: {direction.upper()}\n"
            f"Probabilities → Up {upside_prob}%, Down {downside_prob}%, Flat {flat_prob}%\n"
            f"Entry: {entry:.6f}\n"
            f"SL: {sl:.6f}\n"
            f"TP: {tp_str}\n"
            f"Min RR: {min_rr:.2f}"
        )
        messages.append(msg)

    if messages:
        all_text = "\n\n".join(messages)
        try:
            await context.bot.send_message(chat_id=chat_id, text=all_text)
            job_data["last_signal_time"] = now
            context.job.data = job_data
        except Exception as e:
            logger.exception(f"Error sending autoscan messages: {e}")


async def analyze_coin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Catch-all for commands that are NOT /start /stop /help.
    We treat them as coin commands like:
      /suiusdt
      /suiusdt 4h
    """
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    # Remove leading '/'
    text_no_slash = text.lstrip("/")
    parts = text_no_slash.split()

    if not parts:
        return

    cmd = parts[0]
    tf_arg = parts[1].lower() if len(parts) > 1 else None

    symbol = symbol_from_command(cmd)
    requested_tf = None
    requested_tf_label = None

    if tf_arg and tf_arg in TIMEFRAME_MAP:
        requested_tf, requested_tf_label = TIMEFRAME_MAP[tf_arg]

    await update.message.reply_text(
        f"🔍 Fetching MEXC data for *{symbol}* "
        f"{'('+requested_tf_label+')' if requested_tf_label else '(multi-timeframe)'} ...",
        parse_mode="Markdown",
    )

    # Fetch OHLCV data
    tf_blocks = {}

    if requested_tf:
        candles = fetch_mexc_klines(symbol, requested_tf, limit=200)
        if not candles:
            await update.message.reply_text("❌ Could not fetch candles for that pair/timeframe.")
            return
        tf_blocks[requested_tf_label] = format_ohlcv_for_prompt(candles, max_rows=150)
    else:
        # Multi-timeframe
        for interval, label in MULTI_TF_DEFAULTS:
            candles = fetch_mexc_klines(symbol, interval, limit=200)
            if candles:
                tf_blocks[label] = format_ohlcv_for_prompt(candles, max_rows=150)

        if not tf_blocks:
            await update.message.reply_text("❌ Could not fetch market data for that pair.")
            return

    contents = build_manual_analysis_prompt(symbol, tf_blocks, requested_tf_label)
    data = call_gemini_for_json(contents)
    if not data:
        await update.message.reply_text("❌ Gemini could not produce a valid analysis this time. Try again.")
        return

    upside_prob = int(data.get("upside_prob", 0))
    downside_prob = int(data.get("downside_prob", 0))
    flat_prob = int(data.get("flat_prob", 0))
    dominant = data.get("dominant_scenario", "unknown")
    trade_plan = data.get("trade_plan")
    summary = data.get("summary", "").strip()

    highest = max(upside_prob, downside_prob, flat_prob)

    # Build response text
    lines = []
    lines.append(f"📊 *{symbol}* analysis")
    if requested_tf_label:
        lines.append(f"Timeframe: *{requested_tf_label}*")
    else:
        lines.append("Timeframe: *Multi-timeframe (5m → 1W)*")

    lines.append("")
    lines.append(f"Upside: {upside_prob}%")
    lines.append(f"Downside: {downside_prob}%")
    lines.append(f"Flat/Choppy: {flat_prob}%")
    lines.append(f"Dominant scenario: *{dominant.upper()}*")
    lines.append("")

    if trade_plan and highest >= 75 and dominant in ("upside", "downside"):
        direction = trade_plan.get("direction")
        entry = trade_plan.get("entry")
        sl = trade_plan.get("stop_loss")
        tps = trade_plan.get("take_profits") or []
        min_rr = trade_plan.get("min_rr", 0.0)

        if direction and entry is not None and sl is not None and tps:
            tp_str = ", ".join(f"{tp:.6f}" for tp in tps)
            lines.append("🎯 *Trade idea* (for study, not financial advice):")
            lines.append(f"- Direction: *{direction.upper()}*")
            lines.append(f"- Entry: `{entry:.6f}`")
            lines.append(f"- Stop-loss (beyond key S/R): `{sl:.6f}`")
            lines.append(f"- Take profits: `{tp_str}`")
            lines.append(f"- Min RR: `{min_rr:.2f}`")
            lines.append("")
        else:
            lines.append("No clean trade plan extracted from the model.")
    else:
        lines.append(
            "⚠️ No trade plan suggested (edge < 75% or market too choppy). "
            "Consider waiting for cleaner structure."
        )
        lines.append("")

    if summary:
        lines.append(f"🧠 Summary: {summary}")

    lines.append("")
    lines.append("⚠️ This is *decision support*, not guaranteed profit. Manage risk and size carefully.")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(msg="Exception while handling update:", exc_info=context.error)


# ============================================================
# MAIN
# ============================================================

def main():
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Core commands
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("stop", stop_command))
    application.add_handler(CommandHandler("help", help_command))

    # Any other command → treat as /coin [tf]
    application.add_handler(
        MessageHandler(
            filters.COMMAND & ~filters.Regex(r"^/(start|stop|help)$"),
            analyze_coin_command,
        )
    )

    application.add_error_handler(error_handler)

    logger.info("Bot starting with polling...")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
