# Deploy — launchd

Cópia versionada do que roda de verdade em produção neste Mac. Antes destes
arquivos existirem só em `~/Library/LaunchAgents/`, sem versionamento nem
backup — qualquer edição manual acidental (ou perda do disco) apagava o
histórico de como o sistema estava configurado (P2-40 da auditoria de
2026-09-03).

| Job | Intervalo | O que roda |
|---|---|---|
| `com.caio.polymarket-cycle-light` | 30 min | `run_cycle.py` |
| `com.caio.polymarket-cycle-full` | 4 h | `run_cycle.py --full --html-report` |
| `com.caio.polymarket-ws-feed` | contínuo (`KeepAlive`) | `ws_feed.py` |

Os plists têm o path absoluto (`/Users/caio/faculdade/projects/polymarket`,
`/Users/caio/.local/bin/uv`) fixado — são a config real desta máquina, não
um template. Numa instalação nova, copie pra `~/Library/LaunchAgents/`,
ajuste os paths pro usuário/máquina de destino e rode:

```bash
launchctl load ~/Library/LaunchAgents/com.caio.polymarket-cycle-light.plist
launchctl load ~/Library/LaunchAgents/com.caio.polymarket-cycle-full.plist
launchctl load ~/Library/LaunchAgents/com.caio.polymarket-ws-feed.plist
```

Pra atualizar depois de editar um plist já carregado:

```bash
launchctl unload ~/Library/LaunchAgents/<label>.plist
launchctl load ~/Library/LaunchAgents/<label>.plist
```

Havia também uma entrada de crontab (`0 */6 * * * ... run_cycle.py`) que
nunca chegou a rodar de verdade — o `cron` não tem `uv` no PATH, então toda
execução falhava silenciosamente com `uv: command not found` (11 MB de
`logs/cycle.log` só com essa linha repetida). Removida: se alguém corrigisse
o PATH do cron sem perceber essa entrada, ela passaria a rodar `run_cycle.py`
em paralelo aos jobs do launchd — e `run_cycle.py` não tem lock contra
execução concorrente consigo mesmo fora do próprio flock interno por ciclo.
