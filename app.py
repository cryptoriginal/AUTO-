# ================================
# GEMINI + MEXC CRYPTO ANALYSIS BOT
# Autoscan (Volume spike + Fixed Range Volume Profile)
# Manual Multi-TF Analysis
# Optional BingX Auto-Trade (Market Orders, 7x)
#
# FULLY UPDATED:
# - Strong JSON cleaning
# - Lower temperature (consistent signals)
# - Independent natural probabilities
# - Autoscan logic improved
# - FRVP + Volume spike only
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
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ============================================================
# ENV
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Telegram token missing!")

GEMINI_KEY = (
    os.getenv("GEMINI_API_KEY", "").strip()
    or os.getenv("GOOGLE_API_KEY", "").strip()
)
if not GEMINI_KEY:
    raise RuntimeError("Gemini API key missing!")

gemini_client = genai.Client(api_key=GEMINI_KEY)

# MEXC
MEXC_URL = "https://contract.mexc.com"

# Autoscan params
SCAN_INTERVAL = 300
COOLDOWN = 600
MAX_COINS = 20
AUTOSCAN = {}

# Manual TF
TIMEFRAME_MAP = {
    "5m": ("Min5", "5m"),
    "15m": ("Min15", "15m"),
    "1h": ("Min60", "1h"),
    "2h": ("Min60", "2h"),
    "4h": ("Hour4", "4h"),
    "6h": ("Hour4", "6h"),
    "12h": ("Hour8", "12h"),
    "1d": ("Day1", "1D"),
    "1w": ("Week1", "1W"),
}
MULTI_TF = [
    ("Min5", "5m"),
    ("Min15", "15m"),
    ("Min60", "1h"),
    ("Hour4", "4h"),
    ("Day1", "1D"),
]

# BingX Auto Trade
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
# Helpers
# ============================================================

def symbol_format(cmd: str) -> str:
    cmd = cmd.strip().lstrip("/").upper()
    if cmd.endswith("USDT"):
        return cmd[:-4] + "_USDT"
    if "_" in cmd:
        return cmd
    return cmd + "_USDT"


def get_mexc_candles(symbol: str, interval: str = "Min5", limit: int = 120):
    try:
        url = f"{MEXC_URL}/api/v1/contract/kline/{symbol}"
        now = int(time.time())

        interval_sec = {
            "Min1": 60, "Min5": 300, "Min15": 900,
            "Min60": 3600, "Hour4": 14400,
            "Day1": 86400, "Week1": 604800
        }.get(interval, 300)

        start_ts = now - interval_sec * limit
        r = requests.get(url, params={"interval": interval, "start": start_ts, "end": now}, timeout=8).json()

        if not r.get("success"):
            return []

        d = r["data"]
        out = []
        for i in range(len(d["time"])):
            out.append({
                "time": int(d["time"][i]),
                "open": float(d["open"][i]),
                "high": float(d["high"][i]),
                "low": float(d["low"][i]),
                "close": float(d["close"][i]),
                "volume": float(d["vol"][i]),
            })
        return out
    except:
        return []


def format_candles_for_ai(candles):
    candles = candles[-120:]
    lines = ["time,open,high,low,close,volume"]
    for c in candles:
        ts = datetime.fromtimestamp(c["time"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        lines.append(f"{ts},{c['open']},{c['high']},{c['low']},{c['close']},{c['volume']}")
    return "\n".join(lines)


def get_high_volume_coins(min_vol=50_000_000):
    try:
        r = requests.get(f"{MEXC_URL}/api/v1/contract/ticker", timeout=10).json()
        coins = []
        for x in r.get("data", []):
            if x["symbol"].endswith("_USDT") and float(x.get("amount24", 0)) >= min_vol:
                coins.append((x["symbol"], float(x["amount24"])))
        coins.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in coins[:MAX_COINS]]
    except:
        return []


# ============================================================
# Strong JSON extractor for Gemini
# ============================================================

def extract_clean_json(raw: str):
    """
    Extract JSON between first { and last }
    Works even if Gemini adds text/noise.
    """
    if "{" not in raw or "}" not in raw:
        return None
    raw = raw.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    try:
        return json.loads(raw[start:end+1])
    except:
        return None


def ask_gemini(contents):
    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            generation_config={"temperature": 0.15, "top_p": 0.8},
        )
        raw = resp.text.strip()
        parsed = extract_clean_json(raw)
        if parsed is None:
            log.error(f"Gemini returned invalid JSON:\n{raw}")
        return parsed
    except Exception as e:
        log.error(f"Gemini call failed: {e}")
        return None


# ============================================================
# Autoscan Prompt
# ============================================================

def build_autoscan_prompt(symbol, csv):
    return [
f"""
You are a crypto scalping expert.

TIMEFRAME: 5m
EXCHANGE: MEXC Futures
CONTRACT: {symbol}

ANALYSIS PRIORITY (very important):
1. Sudden volume spike compared to last 20–30 candles
2. Fixed-range volume profile between recent swing high & swing low
3. Price action + candle confirmation (engulfing, hammer, strong wick)
4. Trend structure (HH/HL or LH/LL)

TASK:
Estimate natural probabilities (0–100%), do NOT force 82%:
- upside_prob
- downside_prob
- flat_prob

TRADE RULE:
- If highest_prob < 82 → trade_plan = null
- If highest_prob >= 82:
    * If upside → LONG
    * If downside → SHORT
    * Entry = logical retest or breakout level supported by FRVP + PA
    * SL = beyond key swing (avoid stop hunts)
    * TP = 1–2 levels
    * min_rr >= 1.9

RETURN ONLY JSON:
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
f"CSV DATA:\n{csv}"
]


# ============================================================
# Manual Analysis Prompt
# ============================================================

def build_manual_prompt(symbol, blocks, tf):
    header = (
        f"Focus ONLY on timeframe {tf}."
        if tf else
        "Use MULTI-TIMEFRAME: higher TF = bias, lower TF = entry timing."
    )

    sections = []
    for label, csv in blocks.items():
        sections.append(f"\n=== {label} ===\n{csv}")

    return [
f"""
You are a world-class crypto trader (top 1%).

PAIR: {symbol}

{header}

TASK:
Give NATURAL probabilities (not fixed 75):
- upside_prob
- downside_prob
- flat_prob

RULES:
- If highest_prob < 75 → trade_plan = null (avoid)
- If highest_prob >= 75:
    * direction = long/short
    * entry = logical retest or SR
    * stop_loss = beyond key swing
    * take_profits = 1–2 levels
    * min_rr >= 1.9

RETURN JSON ONLY:
{{
 "upside_prob": int,
 "downside_prob": int,
 "flat_prob": int,
 "dominant_scenario": "upside"|"downside"|"flat",
 "trade_plan": null | {{
    "direction": "long"|"short",
    "entry": float,
    "stop_loss": float,
    "take_profits": [float],
    "min_rr": float
 }},
 "summary": "short reason"
}}
""",
"\n".join(sections)
]


# ============================================================
# BingX Trading
# ============================================================

def mexc_to_bingx(symbol):
    return symbol.replace("_", "-")


def bingx_sign(params):
    qs = "&".join(f"{k}={params[k]}" for k in sorted(params))
    sig = hmac.new(BINGX_API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    return f"{qs}&signature={sig}"


def bingx_price(sym):
    try:
        r = requests.get(f"{BINGX_BASE_URL}/openApi/swap/v2/quote/price", params={"symbol": sym}, timeout=8).json()
        if r.get("success"):
            return float(r["data"]["price"])
    except:
        return None
    return None


def bingx_market_order(symbol, direction):
    if not (BINGX_API_KEY and BINGX_API_SECRET):
        return None

    price = bingx_price(symbol)
    if not price:
        return None

    qty = (BINGX_TRADE_COST_USDT * BINGX_LEVERAGE) / price
    qty = float(f"{qty:.6f}")

    params = {
        "symbol": symbol,
        "side": "BUY" if direction == "long" else "SELL",
        "positionSide": "LONG" if direction == "long" else "SHORT",
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
        log.error(f"BingX order error: {e}")
        return None


# ============================================================
# Autoscan worker
# ============================================================

async def autoscan_job(context):
    d = context.job.data
    chat = d["chat"]
    now = time.time()

    if now - d["last"] < COOLDOWN:
        return

    coins = get_high_volume_coins()
    if not coins:
        return

    out = []

    for coin in coins:
        candles = get_mexc_candles(coin, "Min5", 120)
        if len(candles) < 40:
            continue

        csv = format_candles_for_ai(candles)
        data = ask_gemini(build_autoscan_prompt(coin, csv))
        if not data:
            continue

        up, dn, fl = data["upside_prob"], data["downside_prob"], data["flat_prob"]
        dom = data["dominant_scenario"]
        plan = data["trade_plan"]

        if max(up, dn, fl) < 82 or plan is None:
            continue

        direction = plan["direction"]
        entry = plan["entry"]
        sl = plan["stop_loss"]
        tps = plan["take_profits"]

        # optional auto-trade
        tr = ""
        if BINGX_ENABLE_AUTOTRADE:
            bx = mexc_to_bingx(coin)
            res = bingx_market_order(bx, direction)
            tr = "\nAuto-trade: " + ("✅ Order placed" if (res and res.get("success")) else "❌ Failed")

        out.append(
            f"📡 AUTO SIGNAL\n{coin}\n"
            f"Scenario: {dom.upper()} (Up {up}% / Down {dn}% / Flat {fl}%)\n"
            f"Direction: {direction.upper()}\n"
            f"Entry: {entry}\nSL: {sl}\nTPs: {tps}{tr}"
        )

    if out:
        d["last"] = now
        await context.bot.send_message(chat, "\n\n".join(out))


# ============================================================
# Manual Analysis
# ============================================================

async def manual_analyze(update, context):
    msg = update.message.text.strip().lower()
    parts = msg.lstrip("/").split()

    coin = parts[0]
    symbol = symbol_format(coin)

    tf = None
    if len(parts) > 1 and parts[1] in TIMEFRAME_MAP:
        tf = parts[1]

    await update.message.reply_text("🔎 Analysing...")

    blocks = {}

    if tf:
        interval, label = TIMEFRAME_MAP[tf]
        candles = get_mexc_candles(symbol, interval, 120)
        if not candles:
            return await update.message.reply_text("❌ Bad timeframe or data unavailable.")
        blocks[label] = format_candles_for_ai(candles)
    else:
        for interval, label in MULTI_TF:
            c = get_mexc_candles(symbol, interval, 120)
            if c:
                blocks[label] = format_candles_for_ai(c)

    data = ask_gemini(build_manual_prompt(symbol, blocks, tf))
    if not data:
        return await update.message.reply_text("❌ Gemini could not produce valid JSON. Try again.")

    up, dn, fl = data["upside_prob"], data["downside_prob"], data["flat_prob"]
    dom = data["dominant_scenario"]
    plan = data["trade_plan"]
    summ = data["summary"]

    msg = (
        f"📊 *{symbol}*\n"
        f"Upside: {up}%\nDownside: {dn}%\nFlat: {fl}%\n"
        f"Dominant: *{dom.upper()}*\n\n"
    )

    if plan and max(up, dn, fl) >= 75:
        msg += (
            f"🎯 *Trade Idea*\n"
            f"Direction: {plan['direction']}\n"
            f"Entry: {plan['entry']}\n"
            f"SL: {plan['stop_loss']}\n"
            f"TPs: {plan['take_profits']}\n"
            f"RR: {plan['min_rr']}\n\n"
        )
    else:
        msg += "⚠️ Probability < 75% → Avoid trade.\n\n"

    msg += f"🧠 {summ}"

    await update.message.reply_text(msg, parse_mode="Markdown")


# ============================================================
# Commands
# ============================================================

async def cmd_start(update, context):
    chat = update.effective_chat.id

    await update.message.reply_text(
        "🚀 Auto-scan started.\n"
        "- Volume spike + FRVP + PA confirm\n"
        "- Probability ≥ 82%\n"
        f"- Auto-trade BingX: {'ON' if BINGX_ENABLE_AUTOTRADE else 'OFF'}"
    )

    if chat in AUTOSCAN:
        AUTOSCAN[chat].schedule_removal()

    job = context.job_queue.run_repeating(
        autoscan_job, interval=SCAN_INTERVAL, first=5,
        data={"chat": chat, "last": 0}
    )
    AUTOSCAN[chat] = job


async def cmd_stop(update, context):
    chat = update.effective_chat.id
    if chat not in AUTOSCAN:
        return await update.message.reply_text("Not running.")
    AUTOSCAN[chat].schedule_removal()
    del AUTOSCAN[chat]
    await update.message.reply_text("🛑 Stopped autoscan.")


async def cmd_help(update, context):
    await update.message.reply_text(
        "/start → start autoscan\n"
        "/stop → stop autoscan\n"
        "/btcusdt → multi-TF analysis\n"
        "/suiusdt 4h → TF-specific analysis\n"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("help", cmd_help))

    app.add_handler(MessageHandler(filters.COMMAND, manual_analyze))

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
