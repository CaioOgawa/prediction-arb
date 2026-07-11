"""
notify.py
Alertas via Telegram para o sistema Polymarket Quant.

Configuração (.env):
    TELEGRAM_BOT_TOKEN=123456:ABC...
    TELEGRAM_CHAT_ID=-100123456789   (chat_id do grupo ou seu ID pessoal)

Se as variáveis não estiverem configuradas, as chamadas são silenciosamente ignoradas.
Obtenha o token via @BotFather e o chat_id via @userinfobot ou getUpdates.

Uso:
    from notify import alert, daily_summary
    alert("erro crítico no pipeline")
    daily_summary(pnl=12.50, open_positions=3, win_rate=0.6)
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
_API_URL   = f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage"

HEARTBEAT_FILE = Path("data/heartbeat.json")


def _send(text: str) -> bool:
    """Envia mensagem via Telegram. Retorna True se enviado."""
    if not _BOT_TOKEN or not _CHAT_ID:
        logger.debug("Telegram não configurado (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID ausentes) — skip")
        return False
    try:
        resp = requests.post(
            _API_URL,
            json={"chat_id": _CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        if not resp.ok:
            logger.warning(f"Telegram erro {resp.status_code}: {resp.text[:200]}")
            return False
        return True
    except Exception as e:
        logger.warning(f"Telegram falhou: {e}")
        return False


def alert(message: str, cycle: str = "") -> None:
    """Alerta de falha/erro no ciclo."""
    label = f"[{cycle}] " if cycle else ""
    now   = datetime.now(timezone.utc).strftime("%H:%M UTC")
    text  = f"🚨 <b>Polymarket Quant — {label}ERRO</b>\n{now}\n\n{message}"
    _send(text)


def cycle_ok(mode: str, elapsed_s: float, n_signals: int = 0, n_positions: int = 0) -> None:
    """Confirmação de ciclo bem-sucedido (enviado apenas uma vez por dia, no primeiro ciclo)."""
    now = datetime.now(timezone.utc)
    # Só envia confirmação no primeiro ciclo do dia (hora < 1h)
    if now.hour != 0:
        return
    text = (
        f"✅ <b>Polymarket Quant — Ciclo {mode} OK</b>\n"
        f"{now.strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"⏱ {elapsed_s:.0f}s | 📡 {n_signals} sinais | 📂 {n_positions} posições abertas"
    )
    _send(text)


def daily_summary(
    pnl_usdc: float,
    pnl_pct: float,
    open_positions: int,
    cash: float,
    win_rate: float | None = None,
) -> None:
    """Resumo diário de P&L — enviado uma vez por dia."""
    now    = datetime.now(timezone.utc)
    arrow  = "📈" if pnl_usdc >= 0 else "📉"
    sign   = "+" if pnl_usdc >= 0 else ""
    wr_str = f"{win_rate:.1%}" if win_rate is not None else "—"
    text = (
        f"{arrow} <b>Polymarket Quant — Resumo {now.strftime('%Y-%m-%d')}</b>\n\n"
        f"💵 P&L realizado: <b>{sign}${pnl_usdc:.2f} ({sign}{pnl_pct:.2f}%)</b>\n"
        f"💰 Caixa: ${cash:.2f}\n"
        f"📂 Posições abertas: {open_positions}\n"
        f"🎯 Win rate: {wr_str}"
    )
    _send(text)


ARB_ALERT_STATE      = Path("data/arb_alert_state.json")
ARB_ALERT_COOLDOWN_H = 6  # re-alerta o mesmo arb_group só depois disso


def arb_alert(opps: list[dict]) -> bool:
    """
    Alerta de oportunidades de arb estrutural GARANTIDAS (scanner Fase 2).

    Dedupe por arb_group com cooldown: o ciclo roda a cada 30min e uma
    oportunidade persistente (ex: book ilíquido que ninguém arbitra) não pode
    virar spam. Estado em data/arb_alert_state.json — só atualizado quando o
    envio de fato acontece, para re-tentar se o Telegram falhou.

    Retorna True se um alerta foi enviado.
    """
    import json

    guaranteed = [o for o in opps if o.get("guaranteed")]
    if not guaranteed:
        return False

    now_ts = datetime.now(timezone.utc).timestamp()
    state: dict = {}
    if ARB_ALERT_STATE.exists():
        try:
            state = json.loads(ARB_ALERT_STATE.read_text())
        except Exception:
            state = {}

    fresh = [
        o for o in guaranteed
        if now_ts - float(state.get(str(o.get("arb_group", "")), 0)) > ARB_ALERT_COOLDOWN_H * 3600
    ]
    if not fresh:
        logger.debug(f"arb_alert: {len(guaranteed)} oportunidades, todas em cooldown — skip")
        return False

    lines = []
    for o in sorted(fresh, key=lambda x: -x.get("edge", 0))[:8]:
        lines.append(
            f"• <b>{o.get('kind', '?')}</b> {o.get('arb_group', '')}\n"
            f"  {o.get('n_legs', '?')} pernas | custo {o.get('basket_cost', 0):.3f}"
            f" → payout {o.get('payout_min', 0):.0f}"
            f" | lucro <b>${o.get('profit', 0):.3f} ({o.get('edge', 0):.1%})</b>"
        )
    now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
    text = (
        f"💰 <b>Polymarket Quant — ARB GARANTIDO</b>\n{now_str}\n\n"
        + "\n".join(lines)
        + f"\n\n📂 O paper trader executa baskets garantidos automaticamente no próximo ciclo."
    )

    if not _send(text):
        return False

    for o in fresh:
        state[str(o.get("arb_group", ""))] = now_ts
    try:
        ARB_ALERT_STATE.parent.mkdir(parents=True, exist_ok=True)
        ARB_ALERT_STATE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        logger.warning(f"arb_alert: falha ao salvar estado de dedupe: {e}")
    return True


def write_heartbeat(failures: list[str]) -> None:
    """
    Escreve arquivo data/heartbeat.json com timestamp do último ciclo.
    Usado por scripts de health check externos para detectar se o sistema parou.
    """
    import json
    HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    data = {
        "last_cycle_at":  now.isoformat(),
        "last_cycle_ts":  int(now.timestamp()),
        "failures":       failures,
        "status":         "ok" if not failures else "degraded",
    }
    HEARTBEAT_FILE.write_text(json.dumps(data, indent=2))
    logger.debug(f"Heartbeat escrito: {HEARTBEAT_FILE}")


def check_heartbeat(max_age_minutes: int = 120) -> tuple[bool, str]:
    """
    Lê o heartbeat e verifica se o sistema está vivo.
    Retorna (alive, status_message).
    """
    import json
    if not HEARTBEAT_FILE.exists():
        return False, "heartbeat.json não encontrado — sistema pode nunca ter rodado"
    try:
        data = json.loads(HEARTBEAT_FILE.read_text())
        last_ts = data["last_cycle_ts"]
        age_min = (datetime.now(timezone.utc).timestamp() - last_ts) / 60
        if age_min > max_age_minutes:
            return False, f"último ciclo há {age_min:.0f}min (limite: {max_age_minutes}min)"
        return True, f"OK — último ciclo há {age_min:.0f}min, status={data['status']}"
    except Exception as e:
        return False, f"erro ao ler heartbeat: {e}"


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Testa alertas Telegram")
    p.add_argument("--test",      action="store_true", help="Envia mensagem de teste")
    p.add_argument("--heartbeat", action="store_true", help="Verifica heartbeat local")
    args = p.parse_args()

    if args.test:
        ok = _send("🔔 <b>Teste de conexão</b> — Polymarket Quant configurado corretamente!")
        print("Enviado!" if ok else "Falhou (verifique TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID no .env)")
    if args.heartbeat:
        alive, msg = check_heartbeat()
        print(f"{'✅' if alive else '🚨'} {msg}")
