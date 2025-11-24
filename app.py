# ============================================================
# FULL APP.PY — CLEANEST VERSION
# Gemini 1.5 Pro (no 404), Bybit data, BingX autotrade
# Bulletproof JSON + Background Scanner (NO JobQueue)
# ============================================================

import os
import json
import re
import time
import asyncio
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

from bingx.api import BingxAPI


# ============================================================
# ENVIRONMENT
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "0") or "0")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro-latest")

BINGX_API_KEY = os.getenv("BINGX_API_KEY")
BINGX_API_SECRET = os.getenv("BINGX_API_SECRET")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")


# ============================================================
# INIT CLIENTS (NO 404 MODELS)
# ============================================================

client = genai.Client(api_key=GEMINI_API_KEY)

# Safety: ensure model exists → otherwise fallback to flash model
def resolve_model(name):
    try:
        models = client.models.list()
        available = [m.name for m in models]
        if name in available:
            return name
        if "gemini-1.5-pro-latest" in available:
            return "gemini-1.5-pro-latest"
        return "gemini-1.5-flash"
    except:
        return "gemini-1.5-flash"

MODEL = resolve_model(GEMINI_MODEL)
print("USING GEMINI MODEL:", MODEL)


bingx = None
if BINGX_API_KEY and BINGX_API_SECRET:
    try:
        bingx = BingxAPI(BINGX_API_KEY, BINGX_API_SECRET)
    except:
        try:
            bingx = BingxAPI(BINGX_API_KEY, BINGX_API_SECRET, timestamp="local")
        except:
            bingx = None


# ============================================================
# MARKET DATA (Bybit)
# ============================================================

BYBIT = "https://api.bybit.com"

INTERVAL_MAP = {
    "15m": "15",
    "1h": "60",
    "4h": "240",
}


def get_bybit_symbols():
    try:
        r = requests.get(
            f"{BYBIT}/v5/market/tickers",
            params={"category": "linear"},
            timeout=10,
        )
        data = r.json()
        items = data.get("result", {}).get("list", [])
        coins = [
            it["symbol"] for it in items
            if it["symbol"].endswith("USDT")
            and float(it.get("turnover24h") or 0) >= 40_000_000
        ]
        coins.sort(key=lambda x: float(next(
            it.get("turnover24h") for it in items if it["symbol"] == x
        )), reverse=True)
        return coins[:30]
    except:
        return []


def get_candles(symbol: str, tf: str):
    try:
        r = requests.get(
            f"{BYBIT}/v5/market/kline",
            params={
                "category": "linear",
                "symbol": symbol,
                "interval": INTERVAL_MAP[tf],
                "limit": 150,
            },
            timeout=10,
        )
        data = r.json()
        if data.get("retCode") != 0:
            return []

        out = []
        for c in data["result"]["list"]:
            out.append({
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            })
        return list(reversed(out))
    except:
        return []


def get_price(symbol: str):
    try:
        r = requests.get(
            f"{BYBIT}/v5/market/tickers",
            params={"category": "linear", "symbol": symbol},
            timeout=10,
        )
        items = r.json().get("result", {}).get("list", [])
        if not items:
            return None
        return float(items[0]["lastPrice"])
    except:
        return None
# ============================================================
# SCANNER / RISK SETTINGS
# ============================================================

SCAN_INTERVAL_SECONDS = 300          # background scan interval (5 min)
SIGNAL_COOLDOWN_SECONDS = 600        # 10 min cooldown per (symbol, direction)
MIN_PROB_SCAN = 80                   # autoscan probability threshold
MIN_RR = 1.9                         # minimum RR

AUTO_LEVERAGE = 3
AUTO_MAX_POSITIONS = 2

SCAN_ENABLED = True
SUPPORTED_BINGX: set[str] = set()
auto_open_positions: set[str] = set()
last_signal_time: dict[tuple[str, str], float] = {}


# ============================================================
# BINGX HELPERS
# ============================================================

def load_supported_bingx_symbols():
    """
    Fill SUPPORTED_BINGX with symbols like 'BTCUSDT' that exist on BingX.
    """
    global SUPPORTED_BINGX

    if not bingx:
        SUPPORTED_BINGX = set()
        return

    try:
        data = bingx.get_all_contracts()
    except AttributeError:
        # older lib name fallback
        try:
            data = bingx.swap_v2_get_contracts()
        except Exception as e:
            print("load_supported_bingx_symbols error:", e)
            SUPPORTED_BINGX = set()
            return
    except Exception as e:
        print("load_supported_bingx_symbols error:", e)
        SUPPORTED_BINGX = set()
        return

    if isinstance(data, dict):
        contracts = (
            data.get("data", {}).get("contracts")
            or data.get("data")
            or data.get("contracts")
            or []
        )
    else:
        contracts = data or []

    symbols = set()
    for c in contracts:
        if isinstance(c, dict):
            sym = c.get("symbol") or c.get("pair")
        else:
            sym = str(c)
        if not sym:
            continue
        if sym.endswith("-USDT"):
            symbols.add(sym.replace("-USDT", "USDT"))
        elif sym.endswith("USDT"):
            symbols.add(sym)

    SUPPORTED_BINGX = symbols
    print(f"[BINGX] Loaded {len(SUPPORTED_BINGX)} symbols")


def get_bingx_usdt_balance():
    """Return available margin in USDT from BingX perpetual account."""
    if not bingx:
        return None
    try:
        try:
            info = bingx.get_perpetual_balance()
        except AttributeError:
            info = bingx.swap_v2_get_balance()

        bal = (info.get("data") or {}).get("balance") or {}
        avail = bal.get("availableMargin") or bal.get("balance")
        if avail is None:
            return None
        return float(avail)
    except Exception as e:
        print("get_bingx_usdt_balance error:", e)
        return None


# ============================================================
# GEMINI HELPERS — BULLETPROOF JSON/TEXT
# ============================================================

def _force_json(text: str):
    """
    Extract first valid JSON object from text.
    Very defensive: try direct parse, markdown-clean, slice, regex.
    Return {} if everything fails.
    """
    if not text:
        return {}

    # try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    cleaned = re.sub(r"```(?:json)?", "", text)
    cleaned = cleaned.replace("```", "").strip()

    # parse cleaned
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # slice first {...}
    try:
        start = cleaned.index("{")
        end = cleaned.rindex("}")
        return json.loads(cleaned[start:end + 1])
    except Exception:
        pass

    # regex
    try:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass

    return {}


def gemini_json(prompt: str) -> dict:
    """Call Gemini, try to parse JSON. Return {} on failure."""
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.6),
        )
        txt = getattr(resp, "output_text", None) or getattr(resp, "text", None) or ""
        return _force_json(txt)
    except Exception as e:
        print("Gemini JSON error:", e)
        return {}


def gemini_text(prompt: str) -> str:
    """Plain-text call (fallback when JSON unusable)."""
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.7),
        )
        return getattr(resp, "output_text", None) or getattr(resp, "text", None) or ""
    except Exception as e:
        print("Gemini text error:", e)
        return ""


# ============================================================
# PROMPTS
# ============================================================

def build_scan_prompt(symbol, candles, price):
    return f"""
You are an expert crypto futures analyst.

Analyse:
- Symbol: {symbol}
- Current futures price: {price}
- Candles JSON (15m,1h,4h): {json.dumps(candles)}

Focus on:
- Overall trend and key support/resistance
- VWAP behaviour and fixed-range volume profile zones
- Strong reversal candles at important levels

Only return a trade if it is VERY clean and high probability.

Return ONLY valid JSON, no commentary:

{{
  "symbol": "{symbol}",
  "direction": "long" | "short" | "flat",
  "probability": 0,
  "rr": 0.0,
  "entry": 0.0,
  "stop": 0.0,
  "tp1": 0.0,
  "summary": "very short reason"
}}
"""


def build_manual_prompt(symbol, snapshot, price):
    return f"""
You are a world-class crypto futures trader.

Symbol: {symbol}
Current price: {price}
Snapshot (15m,1h,4h candles): {json.dumps(snapshot)}

Tasks:
1. Estimate upside, downside and flat probabilities (0-100 each).
2. Choose "direction": "long", "short" or "flat".
3. ONLY IF chosen direction probability >= 80:
   - give logical entry, stop, tp1, tp2 based on recent swing highs/lows,
     strong support/resistance, and reversal candles.
4. If no clean trade, set entry/stop/tp1/tp2 to null.

Return ONLY valid JSON:

{{
  "symbol": "{symbol}",
  "direction": "long" | "short" | "flat",
  "summary": "short reason",
  "upside": 0,
  "downside": 0,
  "flat": 0,
  "entry": null | 0.0,
  "stop": null | 0.0,
  "tp1": null | 0.0,
  "tp2": null | 0.0
}}
"""


# ============================================================
# MANUAL ANALYSIS  (/btcusdt, /ethusdt ...)
# ============================================================

async def analyze_manual(symbol: str) -> str:
    symbol = symbol.upper()
    price = get_price(symbol)
    if price is None:
        return f"❌ Could not fetch price for {symbol} from Bybit."

    snapshot = {
        "15m": get_candles(symbol, "15m"),
        "1h": get_candles(symbol, "1h"),
        "4h": get_candles(symbol, "4h"),
    }

    prompt = build_manual_prompt(symbol, snapshot, price)
    data = await asyncio.to_thread(gemini_json, prompt)

    # If JSON unusable → fallback plain text explanation
    if not data or not isinstance(data, dict) or "direction" not in data:
        fallback_prompt = f"""
You are a top crypto trader.

Give a short, 3–6 line trading view for {symbol} at price {price}.
Mention:
- Bias: long / short / flat
- Key support and resistance
- Very brief reasoning.

Do NOT return JSON, just text.
"""
        text = await asyncio.to_thread(gemini_text, fallback_prompt)
        if not text:
            return "❌ AI could not generate analysis. Try again."

        return (
            f"📊 *{symbol} Analysis (fallback)*\n"
            f"Price: `{price}`\n\n"
            f"{text}"
        )

    direction = data.get("direction", "flat")
    summary = data.get("summary", "-")
    up = int(data.get("upside", 0))
    down = int(data.get("downside", 0))
    flat = int(data.get("flat", 0))

    lines = [
        f"📊 *{symbol} Analysis*",
        f"Price: `{price}`",
        f"Direction: *{direction}*",
        f"Upside: `{up}%`  Downside: `{down}%`  Flat: `{flat}%`",
        f"Reason: _{summary}_",
    ]

    entry = data.get("entry")
    stop = data.get("stop")
    tp1 = data.get("tp1")
    tp2 = data.get("tp2")

    if entry and stop and tp1:
        lines += [
            "",
            f"Entry: `{entry}`",
            f"SL: `{stop}`",
            f"TP1: `{tp1}`",
        ]
        if tp2:
            lines.append(f"TP2: `{tp2}`")

    return "\n".join(lines)


# ============================================================
# AUTOTRADE (simple)
# ============================================================

def maybe_autotrade(signal: dict, bot):
    if not bingx:
        return

    sym = signal["symbol"]
    direction = signal["direction"]

    if sym not in SUPPORTED_BINGX:
        print(f"{sym} not supported on BingX — skip autotrade.")
        return

    if len(auto_open_positions) >= AUTO_MAX_POSITIONS:
        print("Max positions reached, skip autotrade.")
        return

    balance = get_bingx_usdt_balance()
    if not balance or balance <= 0:
        if OWNER_CHAT_ID:
            bot.send_message(
                OWNER_CHAT_ID,
                "⚠️ AutoTrade skipped: BingX USDT balance unavailable or zero.",
            )
        return

    entry = signal["entry"]
    if not entry or entry <= 0:
        return

    notional = balance * AUTO_LEVERAGE
    qty = round((notional / AUTO_MAX_POSITIONS) / entry, 4)

    bingx_symbol = f"{sym.replace('USDT', '')}-USDT"
    side = "LONG" if direction == "long" else "SHORT"

    try:
        try:
            bingx.open_market_order(
                bingx_symbol,
                side,
                qty,
                tp=str(signal["tp1"]),
                sl=str(signal["stop"]),
            )
        except AttributeError:
            # very old SDK fallback
            bingx.swap_v2_place_order(
                symbol=bingx_symbol,
                side="BUY" if side == "LONG" else "SELL",
                positionSide=side,
                type="MARKET",
                quantity=str(qty),
                takeProfit=str(signal["tp1"]),
                stopLoss=str(signal["stop"]),
            )

        auto_open_positions.add(sym)

        if OWNER_CHAT_ID:
            bot.send_message(
                OWNER_CHAT_ID,
                f"✅ AutoTrade Executed\n"
                f"Symbol: {bingx_symbol}\n"
                f"Side: {side}\n"
                f"Qty: {qty}\n"
                f"Entry: {entry}\n"
                f"SL: {signal['stop']}\n"
                f"TP1: {signal['tp1']}",
            )
    except Exception as e:
        print("Autotrade error:", e)
        if OWNER_CHAT_ID:
            bot.send_message(OWNER_CHAT_ID, f"❌ AutoTrade error: {e}")
# ============================================================
# AUTOSCAN LOGIC
# ============================================================

async def analyze_signal(symbol: str):
    price = get_price(symbol)
    if price is None:
        return None

    candles = {
        "15m": get_candles(symbol, "15m"),
        "1h": get_candles(symbol, "1h"),
        "4h": get_candles(symbol, "4h"),
    }

    prompt = build_scan_prompt(symbol, candles, price)
    data = await asyncio.to_thread(gemini_json, prompt)

    # JSON failed
    if not data or not isinstance(data, dict):
        return None

    # Filters
    if data.get("direction") == "flat":
        return None
    if int(data.get("probability", 0)) < MIN_PROB_SCAN:
        return None
    if float(data.get("rr", 0.0)) < MIN_RR:
        return None

    try:
        entry = float(data["entry"])
        stop = float(data["stop"])
        tp1 = float(data["tp1"])
    except Exception:
        return None

    return {
        "symbol": symbol,
        "direction": data["direction"],
        "probability": int(data["probability"]),
        "entry": entry,
        "stop": stop,
        "tp1": tp1,
        "rr": float(data["rr"]),
        "summary": data.get("summary", ""),
    }


async def scan_once(app):
    """Runs once every SCAN_INTERVAL_SECONDS"""
    if not SCAN_ENABLED or not OWNER_CHAT_ID:
        return

    symbols = await asyncio.to_thread(get_bybit_symbols)
    if not symbols:
        return

    now = time.time()

    for sym in symbols:
        try:
            sig = await analyze_signal(sym)
        except Exception as e:
            print("scan error for", sym, ":", e)
            continue

        if not sig:
            continue

        # cooldown
        key = (sym, sig["direction"])
        last = last_signal_time.get(key)
        if last and now - last < SIGNAL_COOLDOWN_SECONDS:
            continue
        last_signal_time[key] = now

        # warning
        warn = ""
        if sym not in SUPPORTED_BINGX:
            warn = f"ℹ️ {sym} not supported on BingX → autotrade OFF.\n\n"

        msg = (
            warn
            + f"🚨 *AI SIGNAL*\n"
            f"Symbol: `{sym}`\n"
            f"Direction: `{sig['direction']}`\n"
            f"Probability: `{sig['probability']}%`\n"
            f"RR: `{sig['rr']}`\n"
            f"Entry: `{sig['entry']}`\n"
            f"SL: `{sig['stop']}`\n"
            f"TP1: `{sig['tp1']}`\n"
            f"Reason: _{sig['summary']}_"
        )

        await app.bot.send_message(OWNER_CHAT_ID, msg, parse_mode="Markdown")

        # execute autotrade
        maybe_autotrade(sig, app.bot)


async def scanner_loop(app):
    """Background forever loop."""
    await asyncio.sleep(5)
    while True:
        try:
            await scan_once(app)
        except Exception as e:
            print("scanner_loop error:", e)
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)


# ============================================================
# TELEGRAM HANDLERS
# ============================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SCAN_ENABLED
    SCAN_ENABLED = True
    await update.message.reply_text(
        "✅ Auto-scanner ON.\n"
        f"Scanning every {SCAN_INTERVAL_SECONDS//60} minutes."
    )


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SCAN_ENABLED
    SCAN_ENABLED = False
    await update.message.reply_text("⏹ Auto-scanner OFF.")


async def handle_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles commands like /btcusdt /ethusdt etc."""
    if not update.message:
        return

    text = update.message.text.strip()

    # Remove slash
    if text.startswith("/"):
        symbol = text[1:].split()[0].upper()
    else:
        symbol = text.replace("/", "").split()[0].upper()

    if not symbol.endswith("USDT"):
        await update.message.reply_text(
            "Send pair like: `/btcusdt` or `/ethusdt`",
            parse_mode="Markdown",
        )
        return

    await update.message.reply_text(f"⏳ Analysing {symbol}...")

    try:
        result = await analyze_manual(symbol)
    except Exception as e:
        result = f"❌ Error analysing {symbol}: {e}"

    await update.message.reply_markdown(result)


# ============================================================
# POST_INIT + MAIN
# ============================================================

async def post_init(app):
    await asyncio.to_thread(load_supported_bingx_symbols)
    app.create_task(scanner_loop(app))


def main():
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("stop", cmd_stop))

    # All other commands = coin analyse
    application.add_handler(
        MessageHandler(
            filters.COMMAND & ~filters.Regex(r"^/(start|stop)$"),
            handle_pair,
        )
    )

    print("🚀 Bot is LIVE…")
    application.run_polling()


if __name__ == "__main__":
    main()

