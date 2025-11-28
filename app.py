# ===============================================================
# WEBHOOK VERSION — OPENROUTER (CLAUDE 3.1) + MEXC + AUTOSCAN + AUTOSCALP + BINGX
# 100% COMPATIBLE WITH RENDER WEB SERVICE (NOT WORKER)
# ===============================================================

import os
import json
import time
import hmac
import hashlib
import logging
from datetime import datetime, timezone

import requests
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
    CallbackContext,
)

# ===============================================================
# CONFIG
# ===============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN").strip()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/anthropic/claude-3.1-sonnet").strip()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()

BINGX_API_KEY = os.getenv("BINGX_API_KEY")
BINGX_API_SECRET = os.getenv("BINGX_API_SECRET")
BINGX_ENABLE_AUTOTRADE = os.getenv("BINGX_ENABLE_AUTOTRADE", "false").lower() == "true"
BINGX_TRADE_COST_USDT = float(os.getenv("BINGX_TRADE_COST_USDT", "10"))
BINGX_BASE_URL = "https://open-api.bingx.com"
BINGX_LEVERAGE_AUTOSCAN = 7
BINGX_LEVERAGE_AUTOSCALP = 10

MEXC_URL = "https://contract.mexc.com"

WEBHOOK_PATH = "/webhook"
PORT = int(os.getenv("PORT", "10000"))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bot")

MAX_CANDLES = 60

SCAN_INTERVAL = 300
COOLDOWN = 600
MAX_COINS = 20
AUTOSCAN_JOBS = {}

AUTOSCALP_INTERVAL = 60
AUTOSCALP_JOBS = {}
AUTOSCALP_POSITIONS = {}
BOT_PNL = {}

TIMEFRAME_MAP = {
    "5m": ("Min5", "5m"),
    "15m": ("Min15", "15m"),
    "1h": ("Min60", "1h"),
    "4h": ("Hour4", "4h"),
    "1d": ("Day1", "1D")
}

MULTI_TF = [
    ("Min5", "5m"),
    ("Min60", "1h"),
    ("Hour4", "4h"),
    ("Day1", "1D"),
]

TRAIL_ACTIVATION_PCT = 0.5

# ===============================================================
# HELPERS
# ===============================================================

def symbol_format(cmd):
    cmd = cmd.strip().lstrip("/").upper()
    if cmd.endswith("USDT"):
        return cmd[:-4] + "_USDT"
    return cmd + "_USDT"

def get_mexc_candles(symbol, interval, limit=MAX_CANDLES):
    try:
        url = f"{MEXC_URL}/api/v1/contract/kline/{symbol}"
        now = int(time.time())
        interval_sec = {
            "Min1": 60,
            "Min5": 300,
            "Min15": 900,
            "Min60": 3600,
            "Hour4": 14400,
            "Day1": 86400,
        }.get(interval, 60)
        start_ts = now - interval_sec * limit
        r = requests.get(url, params={"interval":interval,"start":start_ts,"end":now}, timeout=8).json()
        if not r.get("success"):
            return []
        d = r["data"]
        rows=[]
        for i in range(len(d["time"])):
            rows.append({
                "time":int(d["time"][i]),
                "open":float(d["open"][i]),
                "high":float(d["high"][i]),
                "low":float(d["low"][i]),
                "close":float(d["close"][i]),
                "volume":float(d["vol"][i])
            })
        return rows
    except:
        return []

def csv_for_ai(candles):
    lines = ["time,open,high,low,close,volume"]
    for c in candles[-MAX_CANDLES:]:
        ts = datetime.fromtimestamp(c["time"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        lines.append(f"{ts},{c['open']},{c['high']},{c['low']},{c['close']},{c['volume']}")
    return "\n".join(lines)

def get_high_volume_coins():
    try:
        r = requests.get(f"{MEXC_URL}/api/v1/contract/ticker", timeout=10).json()
        coins=[]
        for x in r.get("data",[]):
            sym=x.get("symbol","")
            if sym.endswith("_USDT"):
                vol=float(x.get("amount24",0))
                if vol>=50_000_000:
                    coins.append((sym,vol))
        coins.sort(key=lambda x:x[1], reverse=True)
        return [x[0] for x in coins[:MAX_COINS]]
    except:
        return []

# ===============================================================
# OPENROUTER JSON CALL
# ===============================================================

def openrouter_json(system, user):
    try:
        url = f"{OPENROUTER_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": OPENROUTER_MODEL,
            "messages":[
                {"role":"system","content":system},
                {"role":"user","content":user}
            ],
            "response_format":{"type":"json_object"},
            "temperature":0.2
        }
        r = requests.post(url, headers=headers, json=payload, timeout=40)
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        log.error(f"OpenRouter error: {e}")
        return None

# ===============================================================
# BINGX HELPERS
# ===============================================================

def mexc_to_bingx(sym):
    return sym.replace("_","-")

def sign_bingx(params):
    qs="&".join(f"{k}={params[k]}" for k in sorted(params))
    sig=hmac.new(BINGX_API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    return f"{qs}&signature={sig}"

def bingx_price(symbol):
    try:
        r=requests.get(
            f"{BINGX_BASE_URL}/openApi/swap/v2/quote/price",
            params={"symbol":symbol},
            timeout=8
        ).json()
        if r.get("success"):
            return float(r["data"]["price"])
    except:
        pass
    return None

def bingx_open(symbol, direction, lev, cost):
    if not BINGX_ENABLE_AUTOTRADE:
        return {"success":False}

    px = bingx_price(symbol)
    if not px:
        return {"success":False}

    notional = cost*lev
    qty = notional/px
    qty = float(f"{qty:.6f}")

    side="BUY" if direction=="long" else "SELL"
    pos="LONG" if direction=="long" else "SHORT"

    params={
        "symbol":symbol,
        "side":side,
        "positionSide":pos,
        "type":"MARKET",
        "quantity":qty,
        "timestamp":int(time.time()*1000)
    }
    url=f"{BINGX_BASE_URL}/openApi/swap/v2/trade/order?{sign_bingx(params)}"
    headers={"X-BX-APIKEY":BINGX_API_KEY}

    try:
        r=requests.post(url, headers=headers, timeout=10).json()
        if r.get("success"):
            return {"success":True,"qty":qty,"entry":px}
        return {"success":False}
    except:
        return {"success":False}

def bingx_close(symbol, direction, qty):
    side="SELL" if direction=="long" else "BUY"
    pos="LONG" if direction=="long" else "SHORT"

    params={
        "symbol":symbol,
        "side":side,
        "positionSide":pos,
        "type":"MARKET",
        "quantity":qty,
        "timestamp":int(time.time()*1000)
    }
    url=f"{BINGX_BASE_URL}/openApi/swap/v2/trade/order?{sign_bingx(params)}"
    headers={"X-BX-APIKEY":BINGX_API_KEY}

    try:
        r=requests.post(url, headers=headers, timeout=10).json()
        return r.get("success",False)
    except:
        return False

# ===============================================================
# AUTOSCAN JOB
# ===============================================================

async def autoscan_job(context:CallbackContext):
    data=context.job.data
    chat_id=data["chat"]
    last=data.get("last",0)
    now=time.time()
    if now-last<COOLDOWN:
        return

    syms=get_high_volume_coins()
    if not syms:
        return

    msgs=[]

    for sym in syms:
        candles=get_mexc_candles(sym,"Min5")
        if len(candles)<30:
            continue

        csv=csv_for_ai(candles)

        system=f"""
You output this JSON ONLY:
{{
  "upside_prob":int,
  "downside_prob":int,
  "flat_prob":int,
  "dominant_scenario":"upside"|"downside"|"flat",
  "trade_plan":null|{{
     "direction":"long"|"short",
     "entry":float,
     "stop_loss":float,
     "take_profits":[float],
     "min_rr":float
  }}
}}
Only give JSON, no explanation.
"""

        user=f"PAIR:{sym}\nCSV:\n{csv}"

        res=openrouter_json(system,user)
        if not res:
            continue

        up=res["upside_prob"]
        dn=res["downside_prob"]
        fl=res["flat_prob"]
        dom=res["dominant_scenario"]
        plan=res["trade_plan"]
        if max(up,dn,fl)<82 or not plan:
            continue

        direction=plan["direction"]
        entry=plan["entry"]
        sl=plan["stop_loss"]
        tps=plan["take_profits"]

        trade_msg=""
        if BINGX_ENABLE_AUTOTRADE:
            bx=mexc_to_bingx(sym)
            resp=bingx_open(bx, direction, BINGX_LEVERAGE_AUTOSCAN, BINGX_TRADE_COST_USDT)
            if resp["success"]:
                trade_msg="\nAuto-trade: EXECUTED"
            else:
                trade_msg="\nAuto-trade: FAILED"

        msgs.append(
            f"📡 AUTO SIGNAL {sym}\n"
            f"Scenario:{dom.upper()} (Up {up}% / Down {dn}% / Flat {fl}%)\n"
            f"Dir:{direction.upper()}\n"
            f"Entry:{entry}\nSL:{sl}\nTPs:{tps}{trade_msg}"
        )

    if msgs:
        data["last"]=now
        context.job.data=data
        await context.bot.send_message(chat_id, "\n\n".join(msgs))

# ===============================================================
# AUTOSCALP JOB
# ===============================================================

async def autoscalp_job(context:CallbackContext):
    data=context.job.data
    chat_id=data["chat"]

    positions = AUTOSCALP_POSITIONS.setdefault(chat_id,[])
    pnl = BOT_PNL.setdefault(chat_id,0.0)

    # Manage open scalps
    for pos in positions[:]:
        sym=pos["sym"]
        bx=pos["bx"]
        direction=pos["direction"]
        qty=pos["qty"]
        entry=pos["entry"]
        sl=pos["sl"]
        tp=pos["tp"]
        trail=pos["trail"]
        trail_sl=pos["trail_sl"]

        px=bingx_price(bx)
        if not px:
            continue

        hit=None
        exit_price=None

        if direction=="long":
            if not trail and px>=entry*(1+TRAIL_ACTIVATION_PCT/100):
                pos["trail"]=True
                pos["trail_sl"]=entry
                await context.bot.send_message(chat_id,"Trailing activated LONG")
            if px>=tp:
                hit="TP"; exit_price=tp
            elif px<=sl and not trail:
                hit="SL"; exit_price=sl
            elif trail and px<=trail_sl:
                hit="TRAIL_SL"; exit_price=px
        else:
            if not trail and px<=entry*(1-TRAIL_ACTIVATION_PCT/100):
                pos["trail"]=True
                pos["trail_sl"]=entry
                await context.bot.send_message(chat_id,"Trailing activated SHORT")
            if px<=tp:
                hit="TP"; exit_price=tp
            elif px>=sl and not trail:
                hit="SL"; exit_price=sl
            elif trail and px>=trail_sl:
                hit="TRAIL_SL"; exit_price=px

        if hit:
            closed=bingx_close(bx, direction, qty)
            if not closed:
                continue

            profit=(exit_price-entry)*qty if direction=="long" else (entry-exit_price)*qty
            pnl+=profit
            BOT_PNL[chat_id]=pnl

            await context.bot.send_message(
                chat_id,
                f"Autoscalp CLOSED ({hit}) {bx} {direction}\n"
                f"Entry:{entry}\nExit:{exit_price}\nPnL:{profit:.3f}\nCumulative:{pnl:.2f}"
            )
            positions.remove(pos)

    # Open new scalps if <3 positions
    if len(positions)>=3:
        return

    syms=get_high_volume_coins()
    if not syms:
        return

    for sym in syms:
        if len(positions)>=3:
            break

        candles=get_mexc_candles(sym,"Min1")
        if len(candles)<20:
            continue

        csv=csv_for_ai(candles)

        system=f"""
Only return JSON:
{{
 "take_trade":bool,
 "direction":"long"|"short",
 "probability":int,
 "entry":float,
 "stop_loss":float,
 "take_profit":float
}}
"""

        user=f"PAIR:{sym}\n1m CSV:\n{csv}"

        res=openrouter_json(system,user)
        if not res:
            continue

        if not res["take_trade"] or res["probability"]<80:
            continue

        direction=res["direction"]
        entry_ai=res["entry"]
        sl=res["stop_loss"]
        tp=res["take_profit"]

        bx=mexc_to_bingx(sym)
        resp=bingx_open(bx, direction, BINGX_LEVERAGE_AUTOSCALP, BINGX_TRADE_COST_USDT)
        if not resp["success"]:
            continue

        qty=resp["qty"]
        entry_fill=resp["entry"]

        pos={
            "sym":sym,
            "bx":bx,
            "direction":direction,
            "entry":entry_fill,
            "sl":sl,
            "tp":tp,
            "qty":qty,
            "trail":False,
            "trail_sl":sl
        }
        positions.append(pos)

        await context.bot.send_message(
            chat_id,
            f"Autoscalp OPENED {bx} {direction}\nEntry:{entry_fill}\nSL:{sl}\nTP:{tp}"
        )

# ===============================================================
# MANUAL ANALYSIS
# ===============================================================

async def manual(update:Update, context:ContextTypes.DEFAULT_TYPE):
    text=update.message.text.strip()
    parts=text.lstrip("/").split()
    cmd=parts[0].lower()

    if cmd in ("start","stop","autoscalp","help"):
        return

    tf=parts[1].lower() if len(parts)>1 else None
    symbol=symbol_format(cmd)

    await update.message.reply_text("Analysing...")

    tf_blocks={}
    req_tf=None

    if tf and tf in TIMEFRAME_MAP:
        interval,label=TIMEFRAME_MAP[tf]
        req_tf=label
        candles=get_mexc_candles(symbol,interval)
        if not candles:
            await update.message.reply_text("No data.")
            return
        tf_blocks[label]=csv_for_ai(candles)
    else:
        for interval,label in MULTI_TF:
            candles=get_mexc_candles(symbol,interval)
            if candles:
                tf_blocks[label]=csv_for_ai(candles)

    system=f"""
Return ONLY this JSON:
{{
 "upside_prob":int,
 "downside_prob":int,
 "flat_prob":int,
 "dominant_scenario":"upside"|"downside"|"flat",
 "trade_plan":null|{{
    "direction":"long"|"short",
    "entry":float,
    "stop_loss":float,
    "take_profits":[float],
    "min_rr":float
 }},
 "summary":"short text"
}}
"""

    user=f"PAIR:{symbol}\n{tf_blocks}"

    res=openrouter_json(system,user)
    if not res:
        await update.message.reply_text("❌ JSON error.")
        return

    up=res["upside_prob"]
    dn=res["downside_prob"]
    fl=res["flat_prob"]
    dom=res["dominant_scenario"]
    plan=res["trade_plan"]
    summary=res["summary"]

    msg=[
        f"📊 {symbol}",
        f"Upside:{up}%  Down:{dn}%  Flat:{fl}%",
        f"Dominant:{dom.upper()}",
        ""
    ]

    if plan and max(up,dn,fl)>=75:
        msg.append("🎯 TRADE PLAN:")
        msg.append(f"Direction:{plan['direction'].upper()}")
        msg.append(f"Entry:{plan['entry']}")
        msg.append(f"SL:{plan['stop_loss']}")
        msg.append(f"TPs:{plan['take_profits']}")
        msg.append(f"RR:{plan['min_rr']}")
        msg.append("")
    else:
        msg.append("⚠️ No high-probability setup.")

    msg.append(f"🧠 {summary}")

    await update.message.reply_text("\n".join(msg))

# ===============================================================
# COMMANDS
# ===============================================================

async def cmd_start(update:Update, context:CallbackContext):
    chat_id=update.effective_chat.id

    await update.message.reply_text(
        "Autoscan started.\nRuns every 5 minutes.\n"
        f"Auto-trade:{'ON' if BINGX_ENABLE_AUTOTRADE else 'OFF'}"
    )

    if chat_id in AUTOSCAN_JOBS:
        AUTOSCAN_JOBS[chat_id].schedule_removal()

    job=context.job_queue.run_repeating(
        autoscan_job,
        interval=SCAN_INTERVAL,
        first=5,
        data={"chat":chat_id,"last":0}
    )
    AUTOSCAN_JOBS[chat_id]=job

async def cmd_stop(update:Update, context:CallbackContext):
    chat_id=update.effective_chat.id
    job=AUTOSCAN_JOBS.pop(chat_id,None)
    if job:
        job.schedule_removal()
        await update.message.reply_text("Autoscan stopped.")
    else:
        await update.message.reply_text("Not running.")

async def cmd_autoscalp(update:Update, context:CallbackContext):
    chat_id=update.effective_chat.id
    text=(update.message.text or "").lower()

    if "stop" in text:
        job=AUTOSCALP_JOBS.pop(chat_id,None)
        if job:
            job.schedule_removal()
            await update.message.reply_text("Autoscalp STOPPED.")
        else:
            await update.message.reply_text("Autoscalp not running.")
        return

    await update.message.reply_text(
        "Autoscalp STARTED.\nRuns every 1 minute."
    )

    if chat_id in AUTOSCALP_JOBS:
        AUTOSCALP_JOBS[chat_id].schedule_removal()

    job=context.job_queue.run_repeating(
        autoscalp_job,
        interval=AUTOSCALP_INTERVAL,
        first=5,
        data={"chat":chat_id}
    )
    AUTOSCALP_JOBS[chat_id]=job

async def cmd_help(update:Update, context:CallbackContext):
    await update.message.reply_text(
        "/start - autoscan\n"
        "/stop - stop autoscan\n"
        "/autoscalp - start scalping\n"
        "/autoscalp stop - stop scalping\n"
        "/help - help"
    )

# ===============================================================
# MAIN (WEBHOOK MODE)
# ===============================================================

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start",cmd_start))
    app.add_handler(CommandHandler("stop",cmd_stop))
    app.add_handler(CommandHandler("help",cmd_help))
    app.add_handler(CommandHandler("autoscalp",cmd_autoscalp))
    app.add_handler(MessageHandler(filters.COMMAND,manual))

    # --- WEBHOOK SETUP ---
    import asyncio
    async def run():
        url = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}{WEBHOOK_PATH}"
        await app.bot.set_webhook(url)
        log.info(f"Webhook set to: {url}")
        await app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path="webhook"
        )

    asyncio.run(run())

if __name__=="__main__":
    main()
