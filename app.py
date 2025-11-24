# ================================
# GEMINI + MEXC FUTURES TG BOT
# - Autoscan (VWAP + Volume Profile)
# - Manual Analysis (multi-TF)
# For Render Background Worker
# ================================

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
# ENV & CONFIG
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "0") or "0")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

GEMINI_KEY = (
    os.getenv("GEMINI_API_KEY", "").strip()
    or os.getenv("GOOGLE_API_KEY", "").strip()
)

if not GEMINI_KEY:
    raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in environment")

gemini_client = genai.Client(api_key=GEMINI_KEY)

MEXC_URL = "https://contract.mexc.com"
SCAN_INTERVAL = 300       # 5 minutes
COOLDOWN = 600            # 10 minutes cooldown after sending signals
MAX_COINS = 20            # max pairs per scan

# For autoscan jobs per chat
AUTOSCAN = {}

# Timeframe mapping for manual analysis
TIMEFRAME_MAP = {
    "5m": ("Min5", "5m"),
    "15m": ("Min15", "15m"),
    "1h": ("Min60", "1h"),
    "60m": ("Min60", "1h"),
    "2h": ("Min60", "2h (approx, using 1h data)"),
    "4h": ("Hour4", "4h"),
    "6h": ("Hour4", "6h (approx, using 4h data)"),
    "12h": ("Hour8", "12h"),
    "1d": ("Day1", "1D"),
    "d": ("Day1", "1D"),
    "daily": ("Day1", "1D"),
    "1w": ("Week1", "1W"),
    "w": ("Week1", "1W"),
    "weekly": ("Week1", "1W"),
}

# Default multi-TF set for /coin without timeframe
MULTI_TF_DEFAULTS = [
    ("Min5", "5m"),
    ("Min15", "15m"),
    ("Min60", "1h"),
    ("Hour4", "4h"),
    ("Day1", "1D"),
    ("Week1", "1W"),
]

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("bot")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def symbol_format(cmd: str) -> str:
    """
    Convert /suiusdt -> SUI_USDT, /BTC_USDT -> BTC_USDT, etc.
    """
    cmd = cmd.strip().lstrip("/").upper()
    if cmd.endswith("USDT"):
        base = cmd[:-4]
        return f"{base}_USDT"
    if "_" in cmd:
        return cmd
    return f"{cmd}_USDT"


def get_mexc_candles(symbol: str, interval: str = "Min5", limit: int = 150):
    """
    Fetch OHLCV candles from MEXC futures.
    """
    try:
        url = f"{MEXC_URL}/api/v1/contract/kline/{symbol}"

        now = int(time.time())
        interval_sec = {
            "Min1": 60,
            "Min5": 300,
            "Min15": 900,
            "Min60": 3600,
            "Hour4": 14400,
            "Hour8": 28800,
            "Day1": 86400,
            "Week1": 604800,
        }.get(interval, 300)

        start_ts = now - (interval_sec * limit)
        params = {"interval": interval, "start": start_ts, "end": now}

        r = requests.get(url, params=params, timeout=10).json()
        if not r.get("success"):
            log.warning(f"kline error for {symbol} {interval}: {r}")
            return []

        d = r["data"]
        candles = []
        for i in range(len(d["time"])):
            candles.append(
                {
                    "time": int(d["time"][i]),
                    "open": float(d["open"][i]),
                    "high": float(d["high"][i]),
                    "low": float(d["low"][i]),
                    "close": float(d["close"][i]),
                    "volume": float(d["vol"][i]),
                }
            )
        return candles
    except Exception as e:
        log.error(f"get_mexc_candles error: {e}")
        return []


def get_high_volume_coins(min_vol: float = 50_000_000):
    """
    Get USDT futures contracts with 24h turnover >= min_vol.
    """
    try:
        url = f"{MEXC_URL}/api/v1/contract/ticker"
        r = requests.get(url, timeout=10).json()
        items = r.get("data", [])

        coins = []
        for x in items:
            sym = x.get("symbol", "")
            vol = float(x.get("amount24", 0))
            if sym.endswith("_USDT") and vol >= min_vol:
                coins.append((sym, vol))

        coins.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in coins[:MAX_COINS]]
    except Exception as e:
        log.error(f"get_high_volume_coins error: {e}")
        return []


def format_candles_for_ai(candles, max_rows: int = 150):
    """
    Convert candles list to compact CSV for LLM.
    """
    candles = candles[-max_rows:]
    lines = ["time,open,high,low,close,volume"]
    for c in candles:
        t = datetime.fromtimestamp(c["time"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        lines.append(
            f"{t},{c['open']:.6f},{c['high']:.6f},"
            f"{c['low']:.6f},{c['close']:.6f},{c['volume']:.2f}"
        )
    return "\n".join(lines)


# ============================================================
# GEMINI CALL
# ============================================================

def ask_gemini(contents):
    """
    Call Gemini and parse JSON from response.
    """
    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )
        text = resp.text.strip()

        # strip ```json ...``` wrappers if any
        if text.startswith("```"):
            text = text.strip("`")
            if "\n" in text:
                text = text.split("\n", 1)[1]

        return json.loads(text)
    except Exception as e:
        log.error(f"Gemini error or JSON parse error: {e}")
        return None


# ============================================================
# PROMPTS
# ============================================================

def build_autoscan_prompt(symbol: str, csv_block: str):
    return [
        f"""
You are a crypto futures scalping expert focused on VWAP + fixed-range volume profile.

Exchange: MEXC Futures
Contract: {symbol}
Timeframe: 5m

Data:
- OHLCV given as CSV: time,open,high,low,close,volume.
- Approximate:
  - Session VWAP / intraday VWAP interactions.
  - Fixed-range volume profile between last major swing high & swing low.
  - High-volume nodes / low-volume areas.
- Combine this with candlestick patterns and structure.

Task:
- Estimate probability (0-100%) of:
  1) Upside move (trend up)
  2) Downside move (trend down)
  3) Flat / choppy (avoid trades)

- Pay special attention to:
  - Price reaction at VWAP (bounces, rejections, reclaims).
  - Price at volume profile extremes (HVN/LVN) within last swing.
  - Candle confirmation (wicks, engulfing, hammers, etc).

Trade rules:
- Let highest_prob be max(upside_prob, downside_prob, flat_prob).
- If highest_prob < 85 → trade_plan = null (NO TRADE).
- If highest_prob >= 85:
    - If dominant scenario is upside → LONG.
    - If dominant scenario is downside → SHORT.
    - Entry at logical level aligned with VWAP + volume profile + structure.
    - SL beyond meaningful swing high/low (not tight, avoid stop hunts).
    - 1-2 TP levels, minimum RR >= 1.9.

Output:
Return ONLY valid JSON:

{{
  "upside_prob": int,
  "downside_prob": int,
  "flat_prob": int,
  "dominant_scenario": "upside"|"downside"|"flat",
  "trade_plan": null | {{
     "direction": "long"|"short",
     "entry": float,
     "stop_loss": float,
     "take_profits": [float, ...],
     "min_rr": float
  }}
}}
""",
        f"=== DATA {symbol} 5m ===\n{csv_block}",
    ]


def build_manual_prompt(symbol: str, tf_blocks: dict, requested_tf_label: str | None):
    if requested_tf_label:
        scope = (
            f"Focus ONLY on timeframe: {requested_tf_label}. "
            "Treat it as the main decision frame and ignore other TFs."
        )
    else:
        scope = (
            "Use MULTI-TIMEFRAME analysis: "
            "weekly & daily for bias, 4h/1h for structure, 15m/5m for timing."
        )

    tf_texts = []
    for label, csv_block in tf_blocks.items():
        tf_texts.append(f"=== TF {label} ===\n{csv_block}")

    prompt = f"""
You are a world-class crypto futures trader and risk manager (top 1% globally).

Exchange: MEXC Futures
Contract: {symbol}

{scope}

Goals:
- Estimate probability (0-100%) for next meaningful move:
  1) Upside (sustained bullish move)
  2) Downside (sustained bearish move)
  3) Flat / choppy / range where trade should be avoided.

When reasoning:
- Consider trend, HH/HL / LH/LL, key S/R, liquidity zones.
- Consider candles (hammer, doji, engulfing, pin bars), wicks, rejections.
- Consider basic patterns (flags, wedges, ranges).
- Approximate RSI / MACD / MAs mentally from OHLCV.
- Be conservative → only give trade when edge is clear.

Trade rules:
- Let highest_prob = max(upside_prob, downside_prob, flat_prob).
- If highest_prob < 75 → trade_plan = null.
- If highest_prob >= 75:
    - If dominant scenario is upside → LONG.
    - If dominant scenario is downside → SHORT.
    - Entry near logical level (retest, key S/R, VWAP-like areas).
    - SL beyond meaningful swing high/low (avoid easy stop hunts).
    - 1-2 TP levels with minimum RR >= 1.9.

Output:
Return ONLY valid JSON:

{{
  "upside_prob": int,
  "downside_prob": int,
  "flat_prob": int,
  "dominant_scenario": "upside"|"downside"|"flat",
  "trade_plan": null | {{
     "direction": "long"|"short",
     "entry": float,
     "stop_loss": float,
     "take_profits": [float, ...],
     "min_rr": float
  }},
  "summary": "very short explanation, max 3 sentences"
}}
"""
    return [prompt, "\n\n".join(tf_texts)]


# ============================================================
# COMMAND HANDLERS
# ============================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat.id

    await update.message.reply_text(
        "🚀 Auto-scan started.\n\n"
        "- Every 5 minutes I scan MEXC USDT futures with ≥ 50M 24h volume.\n"
        "- I only send signals when probability ≥ 85% and RR ≥ 1.9.\n"
        "- Focus: VWAP + fixed-range volume profile + candle confluence.\n"
        "- Cooldown ~10 minutes after sending signals.\n\n"
        "Manual analysis still works, e.g. `/btcusdt` or `/suiusdt 4h`."
    )

    # kill old job if exists
    if chat in AUTOSCAN:
        AUTOSCAN[chat].schedule_removal()

    job = context.job_queue.run_repeating(
        autoscan_job,
        interval=SCAN_INTERVAL,
        first=5,
        data={"chat": chat, "last": 0.0},
        name=f"scan_{chat}",
    )
    AUTOSCAN[chat] = job


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat.id

    if chat not in AUTOSCAN:
        await update.message.reply_text("Auto-scan is not running.")
        return

    AUTOSCAN[chat].schedule_removal()
    del AUTOSCAN[chat]
    await update.message.reply_text("🛑 Auto-scan stopped. Manual /coin analysis still available.")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📈 Gemini + MEXC Futures Bot\n\n"
        "Auto-scan:\n"
        "- /start → begin autoscan (5m, VWAP + volume profile)\n"
        "- /stop → stop autoscan\n\n"
        "Manual analysis:\n"
        "- `/btcusdt` → multi-TF (5m→1W) analysis\n"
        "- `/suiusdt 4h` → analysis only on 4h\n"
        "- `/ethusdt 15m` → analysis only on 15m\n\n"
        "I return probabilities for Upside / Downside / Flat.\n"
        "If best scenario ≥ 75% I also suggest entry / SL / TP.\n\n"
        "Use this as decision support, not guaranteed profit. Futures = high risk."
        ,
        parse_mode="Markdown",
    )


# ============================================================
# AUTOSCAN JOB
# ============================================================

async def autoscan_job(context: ContextTypes.DEFAULT_TYPE):
    info = context.job.data
    chat = info["chat"]
    now = time.time()

    # cooldown check
    if now - info["last"] < COOLDOWN:
        return

    coins = get_high_volume_coins()
    if not coins:
        return

    log.info(f"Autoscan {chat}: {coins}")
    messages = []

    for coin in coins:
        candles = get_mexc_candles(coin, "Min5", 150)
        if len(candles) < 50:
            continue

        csv = format_candles_for_ai(candles)
        data = ask_gemini(build_autoscan_prompt(coin, csv))
        if not data:
            continue

        up = int(data.get("upside_prob", 0))
        down = int(data.get("downside_prob", 0))
        flat = int(data.get("flat_prob", 0))
        dom = data.get("dominant_scenario", "flat")
        plan = data.get("trade_plan")

        highest = max(up, down, flat)
        if highest < 85 or not plan:
            continue

        direction = plan.get("direction")
        entry = plan.get("entry")
        sl = plan.get("stop_loss")
        tps = plan.get("take_profits") or []
        rr = plan.get("min_rr", 0.0)

        if not direction or entry is None or sl is None or not tps:
            continue

        tp_str = ", ".join(f"{float(x):.6f}" for x in tps)
        msg = (
            f"📡 AUTO SIGNAL\n"
            f"{coin}\n"
            f"Scenario: {dom.upper()} (Up {up}% / Down {down}% / Flat {flat}%)\n"
            f"Direction: {direction.upper()}\n"
            f"Entry: {float(entry):.6f}\n"
            f"SL: {float(sl):.6f}\n"
            f"TPs: {tp_str}\n"
            f"Min RR: {float(rr):.2f}"
        )
        messages.append(msg)

    if messages:
        all_text = "\n\n".join(messages)
        await context.bot.send_message(chat_id=chat, text=all_text)
        info["last"] = now


# ============================================================
# MANUAL COIN ANALYSIS
# ============================================================

async def coin_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    text_no_slash = text.lstrip("/")
    parts = text_no_slash.split()

    if not parts:
        return

    cmd = parts[0].lower()
    if cmd in ("start", "stop", "help"):
        # already handled by dedicated CommandHandlers
        return

    tf_arg = parts[1].lower() if len(parts) > 1 else None

    symbol = symbol_format(cmd)
    requested_interval = None
    requested_tf_label = None

    if tf_arg and tf_arg in TIMEFRAME_MAP:
        requested_interval, requested_tf_label = TIMEFRAME_MAP[tf_arg]

    await update.message.reply_text(
        f"🔍 Fetching MEXC data for {symbol} "
        f"{'('+requested_tf_label+')' if requested_tf_label else '(multi-timeframe)'} ..."
    )

    tf_blocks = {}

    if requested_interval:
        candles = get_mexc_candles(symbol, requested_interval, 200)
        if not candles:
            await update.message.reply_text("❌ Could not fetch candles for that pair / timeframe.")
            return
        tf_blocks[requested_tf_label] = format_candles_for_ai(candles)
    else:
        # Multi-TF default (option A)
        for interval, label in MULTI_TF_DEFAULTS:
            candles = get_mexc_candles(symbol, interval, 200)
            if candles:
                tf_blocks[label] = format_candles_for_ai(candles)

        if not tf_blocks:
            await update.message.reply_text("❌ Could not fetch any timeframe data for that pair.")
            return

    data = ask_gemini(build_manual_prompt(symbol, tf_blocks, requested_tf_label))
    if not data:
        await update.message.reply_text("❌ Gemini could not produce a valid analysis. Try again.")
        return

    up = int(data.get("upside_prob", 0))
    down = int(data.get("downside_prob", 0))
    flat = int(data.get("flat_prob", 0))
    dom = data.get("dominant_scenario", "flat")
    plan = data.get("trade_plan")
    summary = (data.get("summary") or "").strip()

    highest = max(up, down, flat)

    lines = []
    lines.append(f"📊 *{symbol}* analysis")
    if requested_tf_label:
        lines.append(f"Timeframe: *{requested_tf_label}*")
    else:
        lines.append("Timeframe: *Multi-TF (5m→1W)*")
    lines.append("")
    lines.append(f"Upside: {up}%")
    lines.append(f"Downside: {down}%")
    lines.append(f"Flat/Choppy: {flat}%")
    lines.append(f"Dominant: *{dom.upper()}*")
    lines.append("")

    if plan and highest >= 75 and dom in ("upside", "downside"):
        direction = plan.get("direction")
        entry = plan.get("entry")
        sl = plan.get("stop_loss")
        tps = plan.get("take_profits") or []
        rr = plan.get("min_rr", 0.0)

        if direction and entry is not None and sl is not None and tps:
            tp_str = ", ".join(f"{float(x):.6f}" for x in tps)
            lines.append("🎯 *Trade idea* (for study only, not advice):")
            lines.append(f"- Direction: *{direction.upper()}*")
            lines.append(f"- Entry: `{float(entry):.6f}`")
            lines.append(f"- SL (beyond key S/R): `{float(sl):.6f}`")
            lines.append(f"- TP(s): `{tp_str}`")
            lines.append(f"- Min RR: `{float(rr):.2f}`")
            lines.append("")
        else:
            lines.append("No clean trade plan extracted.")
    else:
        lines.append(
            "⚠️ No trade plan suggested (edge < 75% or market too choppy). "
            "Better to wait for a cleaner setup."
        )
        lines.append("")

    if summary:
        lines.append(f"🧠 Summary: {summary}")
        lines.append("")

    lines.append("⚠️ Use this as decision support, not guaranteed profit. Manage risk & size carefully.")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ============================================================
# MAIN
# ============================================================

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Core commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("help", cmd_help))

    # All other commands → manual analysis
    app.add_handler(MessageHandler(filters.COMMAND, coin_analysis))

    log.info("Bot starting (polling)...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
