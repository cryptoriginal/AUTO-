# FULL APP.PY — STABLE SCANNER + MANUAL ANALYSIS + BINGX AUTOTRADE
# ----------------------------------------------------------------

import os
import json
import time
import asyncio

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
pending_manual_signals = {}  # chat_id -> signal dict

# ============================================================
# CLIENTS
# ============================================================

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

bingx = None
if BINGX_API_KEY and BINGX_API_SECRET:
    # Simple init – matches your installed py-bingx
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

MANUAL_TFS_DEFAULT = ["5m", "15m", "30m", "1h", "4h", "1d"]


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
# GEMINI HELPERS
# ============================================================

def run_gemini(prompt: str) -> str:
    try:
        resp = gemini_model.generate_content(prompt)
        return resp.text or ""
    except Exception as e:
        print("run_gemini error:", e)
        return ""


def extract_json(txt: str):
    if not txt:
        return None
    try:
        s = txt.index("{")
        e = txt.rindex("}")
        return json.loads(txt[s:e+1])
    except Exception as e:
        print("extract_json error:", e, "text:", txt[:200])
        return None


# ============================================================
# SCANNER PROMPT
# ============================================================

def scan_prompt(symbol, candles, price):
    return f"""
You are an expert crypto futures analyst.

Analyze the snapshot:

Symbol: {symbol}
Current price: {price}
Candles: {json.dumps(candles)}

Return ONLY JSON:
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
    tf_text = timeframe or "multi-timeframe (5m to 1D)"

    return f"""
You are a world-class crypto futures trader and risk manager.

Analyse the futures pair {symbol} using the multi-timeframe OHLCV snapshot below.
Key points:
- Timeframes: {", ".join(snapshot.keys())}
- Current futures price: {price}
- Focus on price action, key support/resistance, VWAP behaviour, fixed range volume profile zones,
  and clear reversal candles (hammer, engulfing, doji).

1. Decide whether the next clean move on {tf_text} is:
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

Snapshot (JSON) to analyse:
{json.dumps(snapshot)}
"""


# ============================================================
# BINGX HELPERS (BALANCE & AUTOTRADE)
# ============================================================

def get_bingx_usdt_balance():
    """Return available margin in USDT from BingX perpetual account."""
    if not bingx:
        return None
    try:
        info = bingx.get_perpetual_balance()
        bal = (info.get("data") or {}).get("balance") or {}
        avail = bal.get("availableMargin") or bal.get("balance")
        if avail is None:
            return None
        return float(avail)
    except Exception as e:
        print("get_bingx_usdt_balance error:", e)
        return None


def execute_bingx_trade(signal: dict):
    """
    Synchronous auto-trade execution.
    signal keys: symbol, direction, entry, stop, tp1
    Returns None on success, or error string on failure.
    """
    global auto_open_positions

    if not bingx:
        return "BingX client not configured."

    symbol = signal["symbol"]
    direction = signal["direction"]
    entry = float(signal["entry"])
    stop = float(signal["stop"])
    tp1 = float(signal["tp1"])

    if entry <= 0:
        return "Invalid entry price."

    if len(auto_open_positions) >= AUTO_MAX_POSITIONS:
        return "Max auto positions already open."

    balance = get_bingx_usdt_balance()
    if not balance or balance <= 0:
        return "USDT balance unavailable or zero."

    total_notional = balance * AUTO_LEVERAGE
    qty = (total_notional / AUTO_MAX_POSITIONS) / entry

    side = "LONG" if direction == "long" else "SHORT"
    bingx_symbol = f"{symbol.replace('USDT', '')}-USDT"

    try:
        bingx.open_market_order(
            bingx_symbol,
            side,
            round(qty, 4),
            tp=str(tp1),
            sl=str(stop),
        )
        auto_open_positions.add(symbol)
        return None
    except Exception as e:
        print("execute_bingx_trade error:", e)
        return str(e)


# ============================================================
# MANUAL ANALYSIS
# ============================================================

def build_manual_snapshot(symbol: str, timeframe: str | None):
    if timeframe and timeframe in INTERVAL_MAP:
        tfs = [timeframe]
    else:
        tfs = MANUAL_TFS_DEFAULT

    snapshot = {}
    current_price = None

    for tf in tfs:
        candles = get_candles(symbol, tf)
        snapshot[tf] = candles
        if candles:
            current_price = candles[-1]["close"]

    if current_price is None:
        current_price = get_price(symbol)

    return snapshot, current_price


def analyze_manual(symbol: str, timeframe: str | None):
    """Return (message_text, signal_dict_or_None)."""
    snapshot, price = build_manual_snapshot(symbol, timeframe)
    if price is None:
        return f"❌ Could not fetch candles for {symbol} from Bybit.", None

    prompt = manual_prompt(symbol, timeframe, snapshot, price)
    raw = run_gemini(prompt)
    data = extract_json(raw)
    if not data:
        return "❌ Gemini JSON error.", None

    direction = data.get("direction", "flat")
    up = int(data.get("upside_probability", 0))
    down = int(data.get("downside_probability", 0))
    flat = int(data.get("flat_probability", 0))
    summary = data.get("summary") or "-"

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
    rr = data.get("rr")

    signal = None
    main_prob = max(up, down)

    if (
        direction in ("long", "short")
        and main_prob >= MIN_PROB_SCAN
        and entry is not None
        and stop is not None
        and tp1 is not None
    ):
        lines += [
            "",
            f"Entry: `{entry}`",
            f"SL: `{stop}`",
            f"TP1: `{tp1}`",
        ]
        if tp2:
            lines.append(f"TP2: `{tp2}`")
        if rr:
            lines.append(f"RR: `{rr}`")

        signal = {
            "symbol": symbol,
            "direction": direction,
            "entry": float(entry),
            "stop": float(stop),
            "tp1": float(tp1),
        }
    else:
        lines.append("")
        lines.append(
            f"⛔ Probabilities not strong enough for a clean setup (need ≥ {MIN_PROB_SCAN}%)."
        )

    return "\n".join(lines), signal


# ============================================================
# SCANNER
# ============================================================

async def scan_once(app):
    """Run one full market scan and send signals + auto-trade (scanner)."""
    global last_signal_time

    if not SCAN_ENABLED:
        return
    if OWNER_CHAT_ID == 0:
        return

    symbols = await asyncio.to_thread(get_bybit_symbols)
    if not symbols:
        return

    for sym in symbols:
        c15 = get_candles(sym, "15m")
        c1h = get_candles(sym, "1h")
        c4h = get_candles(sym, "4h")
        price = get_price(sym)
        if price is None:
            continue

        prompt = scan_prompt(sym, {"15m": c15, "1h": c1h, "4h": c4h}, price)
        raw = await asyncio.to_thread(run_gemini, prompt)
        data = extract_json(raw)
        if not data:
            continue

        direction = data.get("direction")
        if direction == "flat":
            continue

        prob = int(data.get("probability", 0))
        rr = float(data.get("rr", 0.0))
        if prob < MIN_PROB_SCAN or rr < MIN_RR:
            continue

        entry = data.get("entry")
        stop = data.get("stop")
        tp1 = data.get("tp1")
        summary = data.get("summary") or "-"
        if entry is None or stop is None or tp1 is None:
            continue

        key = (sym, direction)
        now = time.time()
        last = last_signal_time.get(key)
        if last and now - last < SIGNAL_COOLDOWN_SECONDS:
            continue
        last_signal_time[key] = now

        msg = (
            f"🚨 *AI SIGNAL*\n"
            f"Symbol: `{sym}`\n"
            f"Direction: `{direction}`\n"
            f"Probability: `{prob}%`\n"
            f"RR: `{rr}`\n"
            f"Entry: `{entry}`\n"
            f"SL: `{stop}`\n"
            f"TP1: `{tp1}`\n"
            f"Reason: _{summary}_"
        )

        await app.bot.send_message(OWNER_CHAT_ID, msg, parse_mode="Markdown")

        # Auto-trade for scanner (no confirmation, but respect max positions)
        if bingx and len(auto_open_positions) < AUTO_MAX_POSITIONS:
            signal = {
                "symbol": sym,
                "direction": direction,
                "entry": float(entry),
                "stop": float(stop),
                "tp1": float(tp1),
            }
            err = await asyncio.to_thread(execute_bingx_trade, signal)
            if err:
                await app.bot.send_message(
                    OWNER_CHAT_ID,
                    f"❌ Scanner AutoTrade Error ({sym}): {err}",
                )
            else:
                await app.bot.send_message(
                    OWNER_CHAT_ID,
                    f"✅ Scanner AutoTrade Executed: {sym} {direction} @ {entry}",
                )


async def scanner_loop(app):
    """Background loop that keeps scanning every SCAN_INTERVAL_SECONDS."""
    await asyncio.sleep(5)
    while True:
        try:
            await scan_once(app)
        except Exception as e:
            print("SCAN LOOP ERR:", e)
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)


async def post_init(app):
    """Hook called by PTB after application is initialized."""
    app.create_task(scanner_loop(app))


# ============================================================
# TELEGRAM HANDLERS
# ============================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SCAN_ENABLED
    SCAN_ENABLED = True
    await update.message.reply_text(
        "✅ Auto scanner ON.\n"
        f"Scanning Bybit linear futures every {SCAN_INTERVAL_SECONDS//60} minutes.\n"
        f"Signals only if probability ≥ {MIN_PROB_SCAN}% and RR ≥ {MIN_RR}.\n"
        "Scanner auto-trades instantly (up to 2 positions).\n"
        "Manual analysis can also auto-trade with confirmation if needed."
    )


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SCAN_ENABLED
    SCAN_ENABLED = False
    await update.message.reply_text("⏹ Auto scanner OFF. Manual analysis still works.")


async def handle_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /ETHUSDT or /ETHUSDT 4h etc."""
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
        msg, signal = await asyncio.to_thread(analyze_manual, symbol, timeframe)
    except Exception as e:
        msg = f"❌ Error analysing {symbol}: {e}"
        signal = None

    await update.message.reply_markdown(msg)

    # Autotrade logic for MANUAL analysis
    if (
        signal
        and bingx
        and update.effective_chat.id == OWNER_CHAT_ID
    ):
        await handle_manual_autotrade(signal, context.application, update.effective_chat.id)


async def handle_manual_autotrade(signal: dict, app, chat_id: int):
    """
    Manual autotrade logic:
    - If <2 positions: execute immediately.
    - If >=2 positions: ask for /yes /no confirmation.
    """
    global pending_manual_signals

    if len(auto_open_positions) < AUTO_MAX_POSITIONS:
        err = await asyncio.to_thread(execute_bingx_trade, signal)
        if err:
            await app.bot.send_message(chat_id, f"❌ AutoTrade Error: {err}")
        else:
            await app.bot.send_message(
                chat_id,
                f"✅ Manual AutoTrade Executed: {signal['symbol']} {signal['direction']} @ {signal['entry']}",
            )
    else:
        pending_manual_signals[chat_id] = signal
        await app.bot.send_message(
            chat_id,
            "⚠️ You already have 2 auto positions open.\n"
            "Send /yes to execute this manual trade anyway, or /no to cancel.",
        )


async def cmd_yes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Confirm pending manual trade."""
    chat_id = update.effective_chat.id
    if chat_id != OWNER_CHAT_ID:
        return

    signal = pending_manual_signals.pop(chat_id, None)
    if not signal:
        await update.message.reply_text("No pending manual trade to confirm.")
        return

    err = await asyncio.to_thread(execute_bingx_trade, signal)
    if err:
        await update.message.reply_text(f"❌ AutoTrade Error: {err}")
    else:
        await update.message.reply_text(
            f"✅ Manual trade executed: {signal['symbol']} {signal['direction']} @ {signal['entry']}"
        )


async def cmd_no(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel pending manual trade."""
    chat_id = update.effective_chat.id
    if chat_id != OWNER_CHAT_ID:
        return

    if chat_id in pending_manual_signals:
        pending_manual_signals.pop(chat_id, None)
        await update.message.reply_text("❌ Manual trade cancelled.")
    else:
        await update.message.reply_text("No pending manual trade to cancel.")


# ============================================================
# MAIN
# ============================================================

def main():
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)   # attach scanner loop
        .build()
    )

    # Order matters: specific commands first, then generic /pair
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CommandHandler("yes", cmd_yes))
    application.add_handler(CommandHandler("no", cmd_no))

    # Any other /command = coin/timeframe
    application.add_handler(MessageHandler(filters.COMMAND, handle_pair))

    print("Bot running...")
    application.run_polling()


if __name__ == "__main__":
    main()
