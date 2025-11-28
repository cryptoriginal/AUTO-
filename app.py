# ================================
# GEMINI + MEXC CRYPTO ANALYSIS BOT (Balanced, JSON-Stable)
# - Manual analysis (/btcusdt, /suiusdt 4h, etc.)
# - Autoscan (5m, volume spike + FRVP-style + PA)
# - Optional BingX auto-trade (MARKET, 7x)
# - Uses response_mime_type="application/json" for stable parsing
# ================================

import os
import time
import json
import hmac
import hashlib
import logging
from datetime import datetime, timezone

import requests
from google import genai
from google.genai import types
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ============================================================
# ENV & BASIC CONFIG
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN missing")

GEMINI_KEY = (
    os.getenv("GEMINI_API_KEY", "").strip()
    or os.getenv("GOOGLE_API_KEY", "").strip()
)
if not GEMINI_KEY:
    raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY missing")

gemini_client = genai.Client(api_key=GEMINI_KEY)

MEXC_URL = "https://contract.mexc.com"

# Autoscan config
SCAN_INTERVAL = 300      # 5m
COOLDOWN = 600           # 10m cooldown after signals
MAX_COINS = 20
AUTOSCAN = {}

# Candles per TF (balanced)
MAX_CANDLES = 60

# Manual timeframe map
TIMEFRAME_MAP = {
    "5m": ("Min5", "5m"),
    "15m": ("Min15", "15m"),
    "1h": ("Min60", "1h"),
    "4h": ("Hour4", "4h"),
    "1d": ("Day1", "1D"),
}
# Multi-TF default for manual analysis
MULTI_TF = [
    ("Min5", "5m"),
    ("Min60", "1h"),
    ("Hour4", "4h"),
    ("Day1", "1D"),
]

# BingX auto-trade config
BINGX_API_KEY = os.getenv("BINGX_API_KEY", "").strip()
BINGX_API_SECRET = os.getenv("BINGX_API_SECRET", "").strip()
BINGX_ENABLE_AUTOTRADE = os.getenv("BINGX_ENABLE_AUTOTRADE", "false").lower() == "true"
BINGX_TRADE_COST_USDT = float(os.getenv("BINGX_TRADE_COST_USDT", "10"))
BINGX_BASE_URL = "https://open-api.bingx.com"
BINGX_LEVERAGE = 7

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("bot")


# ============================================================
# BASIC HELPERS (MEXC)
# ============================================================

def symbol_format(cmd: str) -> str:
    cmd = cmd.strip().lstrip("/").upper()
    if cmd.endswith("USDT"):
        return cmd[:-4] + "_USDT"
    if "_" in cmd:
        return cmd
    return cmd + "_USDT"


def get_mexc_candles(symbol: str, interval: str, limit: int = MAX_CANDLES):
    try:
        url = f"{MEXC_URL}/api/v1/contract/kline/{symbol}"
        now = int(time.time())
        interval_sec = {
            "Min1": 60, "Min5": 300, "Min15": 900,
            "Min60": 3600, "Hour4": 14400,
            "Day1": 86400, "Week1": 604800,
        }.get(interval, 300)
        start_ts = now - interval_sec * limit
        r = requests.get(
            url,
            params={"interval": interval, "start": start_ts, "end": now},
            timeout=8,
        ).json()
        if not r.get("success"):
            log.warning(f"MEXC kline error for {symbol} {interval}: {r}")
            return []
        d = r["data"]
        out = []
        for i in range(len(d["time"])):
            out.append(
                {
                    "time": int(d["time"][i]),
                    "open": float(d["open"][i]),
                    "high": float(d["high"][i]),
                    "low": float(d["low"][i]),
                    "close": float(d["close"][i]),
                    "volume": float(d["vol"][i]),
                }
            )
        return out
    except Exception as e:
        log.error(f"get_mexc_candles error: {e}")
        return []


def format_candles_for_ai(candles):
    candles = candles[-MAX_CANDLES:]
    lines = ["time,open,high,low,close,volume"]
    for c in candles:
        ts = datetime.fromtimestamp(c["time"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        lines.append(
            f"{ts},{c['open']},{c['high']},{c['low']},{c['close']},{c['volume']}"
        )
    return "\n".join(lines)


def get_high_volume_coins(min_vol=50_000_000):
    try:
        r = requests.get(f"{MEXC_URL}/api/v1/contract/ticker", timeout=10).json()
        coins = []
        for x in r.get("data", []):
            sym = x.get("symbol", "")
            if not sym.endswith("_USDT"):
                continue
            vol = float(x.get("amount24", 0))
            if vol >= min_vol:
                coins.append((sym, vol))
        coins.sort(key=lambda p: p[1], reverse=True)
        return [s for s, _ in coins[:MAX_COINS]]
    except Exception as e:
        log.error(f"get_high_volume_coins error: {e}")
        return []


# ============================================================
# GEMINI JSON CALL (FORCED JSON OUTPUT)
# ============================================================

def parse_json_text(text: str):
    """
    Handle both:
      - pure JSON: {"a":1}
      - JSON-as-string: "{\"a\":1}"
    """
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, str):
            try:
                return json.loads(obj)
            except Exception:
                return None
        return obj
    except Exception as e:
        log.error(f"json.loads error: {e}")
        return None


def gemini_json(contents):
    """
    Call Gemini with response_mime_type='application/json'
    So resp.text should always be JSON or JSON string.
    """
    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.15,
                top_p=0.8,
            ),
        )
        txt = (resp.text or "").strip()
        obj = parse_json_text(txt)
        if obj is None:
            log.error(f"Gemini JSON parse failed. Raw (truncated): {txt[:500]}")
        return obj
    except Exception as e:
        log.error(f"Gemini API error: {e}")
        return None


# ============================================================
# PROMPTS
# ============================================================

AUTOSCAN_SCHEMA_DESC = """
{
  "upside_prob": int,
  "downside_prob": int,
  "flat_prob": int,
  "dominant_scenario": "upside"|"downside"|"flat",
  "trade_plan": null | {
     "direction": "long"|"short",
     "entry": float,
     "stop_loss": float,
     "take_profits": [float, ...],
     "min_rr": float
  }
}
"""

MANUAL_SCHEMA_DESC = """
{
  "upside_prob": int,
  "downside_prob": int,
  "flat_prob": int,
  "dominant_scenario": "upside"|"downside"|"flat",
  "trade_plan": null | {
     "direction": "long"|"short",
     "entry": float,
     "stop_loss": float,
     "take_profits": [float, ...],
     "min_rr": float
  },
  "summary": "short explanation"
}
"""


def build_autoscan_prompt(symbol: str, csv_block: str):
    return [
f"""
You are a crypto scalper.

PAIR: {symbol}
TIMEFRAME: 5m
DATA: OHLCV CSV is provided separately.

Focus on:
- Sudden volume spikes vs last ~30 candles.
- Fixed-range volume profile between recent swing high & low (approx from OHLCV).
- Price action & candles at these key areas (engulfing, hammers, strong rejections).
- Trend structure (HH/HL vs LH/LL).

Task:
- Estimate realistic probabilities (0-100):
  upside_prob, downside_prob, flat_prob.
- dominant_scenario is whichever has highest probability.

Trade rules:
- highest_prob = max(ups, downs, flat).
- If highest_prob < 82 → trade_plan = null (NO TRADE).
- If highest_prob >= 82:
    - If upside dominant → LONG.
    - If downside dominant → SHORT.
    - entry: logical retest or breakout level aligned with volume spike + PA.
    - stop_loss: beyond recent swing high/low (avoid easy stop hunts).
    - take_profits: 1-2 levels.
    - min_rr >= 1.9.

Output:
Return ONLY ONE JSON object of this form:
{AUTOSCAN_SCHEMA_DESC}
""",
f"CSV_DATA:\n{csv_block}",
]


def build_manual_prompt(symbol: str, tf_blocks: dict, requested_tf: str | None):
    scope = (
        f"Focus ONLY on timeframe {requested_tf}."
        if requested_tf
        else "Use multiple timeframes: higher TF for bias, lower TF for entry timing."
    )

    sections = []
    for label, csv in tf_blocks.items():
        sections.append(f"\n=== {label} ===\n{csv}")

    return [
f"""
You are a world-class crypto futures trader and risk manager.

PAIR: {symbol}

{scope}

From the OHLCV data, estimate natural probabilities (0-100) WITHOUT forcing them to 75:
- upside_prob
- downside_prob
- flat_prob

Guidelines:
- If market is messy or conflicting → increase flat_prob.
- Use trend, S/R, candle patterns, basic chart patterns, rough volume profile.

Trade rules:
- highest_prob = max(ups, downs, flat).
- If highest_prob < 75 → trade_plan = null (avoid trade).
- If highest_prob >= 75:
    - If upside dominant → LONG.
    - If downside dominant → SHORT.
    - entry: logical retest/SR/breakout level.
    - stop_loss: beyond recent key swing.
    - take_profits: 1–2 levels, min_rr >= 1.9.

Output:
Return ONLY ONE JSON object of this form:
{MANUAL_SCHEMA_DESC}
""",
"".join(sections),
]


# ============================================================
# BINGX HELPERS
# ============================================================

def mexc_to_bingx_symbol(sym: str) -> str:
    return sym.replace("_", "-")


def bingx_sign(params: dict) -> str:
    qs = "&".join(f"{k}={params[k]}" for k in sorted(params))
    sig = hmac.new(BINGX_API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    return f"{qs}&signature={sig}"


def bingx_get_price(symbol: str) -> float | None:
    try:
        r = requests.get(
            f"{BINGX_BASE_URL}/openApi/swap/v2/quote/price",
            params={"symbol": symbol},
            timeout=8,
        ).json()
        if r.get("success") and r.get("data"):
            return float(r["data"]["price"])
    except Exception as e:
        log.error(f"bingx_get_price error: {e}")
    return None


def bingx_place_market(symbol: str, direction: str):
    if not (BINGX_API_KEY and BINGX_API_SECRET):
        log.info("BingX keys missing, skipping auto-trade.")
        return None

    price = bingx_get_price(symbol)
    if not price or price <= 0:
        log.warning(f"No BingX price for {symbol}")
        return None

    notional = BINGX_TRADE_COST_USDT * BINGX_LEVERAGE
    qty = notional / price
    qty = float(f"{qty:.6f}")

    side = "BUY" if direction == "long" else "SELL"
    pos_side = "LONG" if direction == "long" else "SHORT"

    params = {
        "symbol": symbol,
        "side": side,
        "positionSide": pos_side,
        "type": "MARKET",
        "quantity": qty,
        "timestamp": int(time.time() * 1000),
    }

    url = f"{BINGX_BASE_URL}/openApi/swap/v2/trade/order?{bingx_sign(params)}"
    headers = {"X-BX-APIKEY": BINGX_API_KEY}
    try:
        r = requests.post(url, headers=headers, timeout=10).json()
        log.info(f"BingX order response: {r}")
        return r
    except Exception as e:
        log.error(f"bingx_place_market error: {e}")
        return None


# ============================================================
# AUTOSCAN JOB
# ============================================================

async def autoscan_job(context: ContextTypes.DEFAULT_TYPE):
    data = context.job.data
    chat_id = data["chat"]
    last_ts = data.get("last", 0.0)
    now = time.time()

    if now - last_ts < COOLDOWN:
        return

    symbols = get_high_volume_coins()
    if not symbols:
        return

    log.info(f"Autoscan symbols: {symbols}")
    messages = []

    for sym in symbols:
        candles = get_mexc_candles(sym, "Min5", MAX_CANDLES)
        if len(candles) < 30:
            continue

        csv = format_candles_for_ai(candles)
        result = gemini_json(build_autoscan_prompt(sym, csv))
        if not result:
            continue

        up = int(result.get("upside_prob", 0))
        dn = int(result.get("downside_prob", 0))
        fl = int(result.get("flat_prob", 0))
        dom = result.get("dominant_scenario", "flat")
        plan = result.get("trade_plan")

        highest = max(up, dn, fl)
        if highest < 82 or not plan:
            continue

        direction = plan.get("direction")
        entry = plan.get("entry")
        sl = plan.get("stop_loss")
        tps = plan.get("take_profits") or []
        rr = float(plan.get("min_rr", 0.0))

        if not direction or entry is None or sl is None or not tps:
            continue

        tp_str = ", ".join(f"{float(x):.6f}" for x in tps)

        trade_info = ""
        if BINGX_ENABLE_AUTOTRADE:
            bx_sym = mexc_to_bingx_symbol(sym)
            resp = bingx_place_market(bx_sym, direction)
            if resp and resp.get("success"):
                trade_info = "\nAuto-trade: ✅ BingX MARKET 7x order placed."
            else:
                trade_info = "\nAuto-trade: ❌ Failed (check logs/API)."

        messages.append(
            f"📡 AUTO SIGNAL\n"
            f"{sym}\n"
            f"Scenario: {dom.upper()} (Up {up}% / Down {dn}% / Flat {fl}%)\n"
            f"Direction: {direction.upper()}\n"
            f"Entry: {float(entry):.6f}\n"
            f"SL: {float(sl):.6f}\n"
            f"TPs: {tp_str}\n"
            f"Min RR: {rr:.2f}"
            f"{trade_info}"
        )

    if messages:
        data["last"] = now
        context.job.data = data
        await context.bot.send_message(chat_id=chat_id, text="\n\n".join(messages))


# ============================================================
# MANUAL ANALYSIS
# ============================================================

async def manual_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    parts = text.lstrip("/").split()
    if not parts:
        return

    cmd = parts[0].lower()
    if cmd in ("start", "stop", "help"):
        return

    tf_arg = parts[1].lower() if len(parts) > 1 else None
    symbol = symbol_format(cmd)

    await update.message.reply_text("🔍 Analysing, please wait...")

    tf_blocks = {}
    requested_tf_label = None

    if tf_arg and tf_arg in TIMEFRAME_MAP:
        interval, label = TIMEFRAME_MAP[tf_arg]
        requested_tf_label = label
        candles = get_mexc_candles(symbol, interval, MAX_CANDLES)
        if not candles:
            await update.message.reply_text("❌ Could not fetch data for that timeframe.")
            return
        tf_blocks[label] = format_candles_for_ai(candles)
    else:
        for interval, label in MULTI_TF:
            candles = get_mexc_candles(symbol, interval, MAX_CANDLES)
            if candles:
                tf_blocks[label] = format_candles_for_ai(candles)
        if not tf_blocks:
            await update.message.reply_text("❌ No data available for this pair.")
            return

    result = gemini_json(build_manual_prompt(symbol, tf_blocks, requested_tf_label))
    if not result:
        await update.message.reply_text("❌ Gemini could not produce valid JSON. Try again.")
        return

    up = int(result.get("upside_prob", 0))
    dn = int(result.get("downside_prob", 0))
    fl = int(result.get("flat_prob", 0))
    dom = result.get("dominant_scenario", "flat")
    plan = result.get("trade_plan")
    summary = (result.get("summary") or "").strip()

    highest = max(up, dn, fl)

    lines = []
    lines.append(f"📊 *{symbol}* analysis")
    if requested_tf_label:
        lines.append(f"Timeframe: *{requested_tf_label}*")
    else:
        lines.append("Timeframe: *Multi-TF (5m, 1h, 4h, 1D)*")
    lines.append("")
    lines.append(f"Upside: {up}%")
    lines.append(f"Downside: {dn}%")
    lines.append(f"Flat/Choppy: {fl}%")
    lines.append(f"Dominant: *{dom.upper()}*")
    lines.append("")

    if plan and highest >= 75 and dom in ("upside", "downside"):
        direction = plan.get("direction")
        entry = plan.get("entry")
        sl = plan.get("stop_loss")
        tps = plan.get("take_profits") or []
        rr = float(plan.get("min_rr", 0.0))

        if direction and entry is not None and sl is not None and tps:
            tp_str = ", ".join(f"{float(x):.6f}" for x in tps)
            lines.append("🎯 *Trade idea* (for study only):")
            lines.append(f"- Direction: *{direction.upper()}*")
            lines.append(f"- Entry: `{float(entry):.6f}`")
            lines.append(f"- SL: `{float(sl):.6f}`")
            lines.append(f"- TP(s): `{tp_str}`")
            lines.append(f"- Min RR: `{rr:.2f}`")
            lines.append("")
        else:
            lines.append("No clean trade plan extracted.")
    else:
        lines.append(
            "⚠️ Highest probability < 75% or market too choppy.\n"
            "➡️ Better to avoid this setup."
        )
        lines.append("")

    if summary:
        lines.append(f"🧠 {summary}")
        lines.append("")

    lines.append("⚠️ Use this as decision support, not guaranteed profit. Manage risk carefully.")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ============================================================
# COMMAND HANDLERS
# ============================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    await update.message.reply_text(
        "🚀 Auto-scan started.\n\n"
        "- Every 5 minutes: scan MEXC USDT futures with 24h volume ≥ 50M.\n"
        "- Logic: volume spike + FRVP-style + price action.\n"
        "- Signals only if probability ≥ 82% and RR ≥ 1.9.\n"
        f"- Auto-trade BingX: {'ON ✅' if BINGX_ENABLE_AUTOTRADE else 'OFF ❌'}"
    )

    if chat_id in AUTOSCAN:
        AUTOSCAN[chat_id].schedule_removal()

    job = context.job_queue.run_repeating(
        autoscan_job,
        interval=SCAN_INTERVAL,
        first=5,
        data={"chat": chat_id, "last": 0.0},
        name=f"autoscan_{chat_id}",
    )
    AUTOSCAN[chat_id] = job


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    job = AUTOSCAN.pop(chat_id, None)
    if job:
        job.schedule_removal()
        await update.message.reply_text("🛑 Auto-scan stopped.")
    else:
        await update.message.reply_text("Auto-scan is not running for this chat.")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📈 Gemini + MEXC Futures Bot (Balanced JSON Mode)\n\n"
        "Auto-scan:\n"
        "- /start → start autoscan (5m, volume spike + FRVP-style)\n"
        "- /stop → stop autoscan\n\n"
        "Manual analysis:\n"
        "- `/btcusdt` → multi-TF (5m, 1h, 4h, 1D)\n"
        "- `/suiusdt 4h` → single TF\n"
        "- `/ethusdt 1h` → single TF\n\n"
        "I only give entry/SL/TP if highest probability ≥ 75%.\n"
        "Below 75%: I’ll tell you to avoid the trade.\n\n"
        f"Auto-trade (BingX MARKET 7x): {'ENABLED' if BINGX_ENABLE_AUTOTRADE else 'DISABLED'}",
        parse_mode="Markdown",
    )


# ============================================================
# MAIN
# ============================================================

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("help", cmd_help))

    # All other commands → manual analysis
    app.add_handler(MessageHandler(filters.COMMAND, manual_analyze))

    log.info("Bot starting (polling)...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

