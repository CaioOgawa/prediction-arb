"""
validate_connection.py
Testa a conectividade com todas as APIs do Polymarket antes de iniciar o projeto.
Execute após configurar o .env para garantir que tudo está funcionando.

Uso:
    python pipeline/validate_connection.py
"""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()


def test_gamma_api() -> bool:
    """Testa Gamma API — não requer autenticação"""
    url = "https://gamma-api.polymarket.com/markets?limit=5&active=true"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        markets = r.json()
        print(f"  [OK] Gamma API — {len(markets)} mercados retornados")
        return True
    except Exception as e:
        print(f"  [ERRO] Gamma API: {e}")
        return False


def test_clob_api_public() -> bool:
    """Testa endpoint público da CLOB API — não requer autenticação"""
    url = "https://clob.polymarket.com/markets?limit=1"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        print("  [OK] CLOB API (público)")
        return True
    except Exception as e:
        print(f"  [ERRO] CLOB API público: {e}")
        return False


def test_clob_api_auth() -> bool:
    """Testa CLOB API autenticada — requer credenciais no .env"""
    api_key = os.getenv("POLY_API_KEY")
    private_key = os.getenv("POLY_PRIVATE_KEY")

    if not api_key or api_key == "seu_api_key":
        print("  [SKIP] CLOB API (auth) — credenciais não configuradas no .env")
        return True  # Não falha, só avisa

    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds
        from py_clob_client.constants import POLYGON

        creds = ApiCreds(
            api_key=os.getenv("POLY_API_KEY"),
            api_secret=os.getenv("POLY_API_SECRET"),
            api_passphrase=os.getenv("POLY_API_PASSPHRASE"),
        )
        client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=POLYGON,
            creds=creds,
        )
        # Endpoint simples para validar auth
        resp = client.get_markets(next_cursor="")
        print("  [OK] CLOB API (autenticada)")
        return True
    except ImportError:
        print("  [SKIP] CLOB API (auth) — py-clob-client não instalado")
        return True
    except Exception as e:
        print(f"  [ERRO] CLOB API autenticada: {e}")
        return False


def test_thegraph() -> bool:
    """
    Testa subgraph Polymarket no TheGraph.
    Nota: o hosted service do TheGraph foi depreciado em 2024.

    P2-42: o retorno reflete o resultado real do teste — antes retornava
    True em todo caminho, inclusive falha, então "todas as conexões
    validadas com sucesso" no __main__ mentia quando o TheGraph estava fora
    do ar. "Não-bloqueante" é decisão de quem chama (__main__ não inclui
    isso no gate de sucesso porque é fonte secundária), não uma mentira
    embutida no valor de retorno.
    """
    # Endpoint do subgraph na rede descentralizada do TheGraph
    url = "https://gateway.thegraph.com/api/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
    query = '{ fixedProductMarketMakers(first: 3) { id collateralVolume } }'
    try:
        r = requests.post(url, json={"query": query}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "errors" in data:
            print(f"  [AVISO] TheGraph retornou erros: {data['errors']}")
            return False
        print("  [OK] TheGraph (rede descentralizada)")
        return True
    except Exception as e:
        # Não-bloqueante: TheGraph é fonte secundária
        print(f"  [AVISO] TheGraph indisponivel (nao-bloqueante): {type(e).__name__}")
        print("           Fonte primaria (Gamma API) esta OK — pode continuar.")
        return False


def check_env_vars() -> None:
    """Informa quais variáveis de ambiente estão configuradas"""
    # P2-42: POLY_PROXY_WALLET/NEWSAPI_KEY nunca tiveram leitor no código —
    # tiradas daqui junto da remoção do .env/.env.example. ODDS_API_KEY é a
    # única credencial cuja ausência de fato levanta exceção
    # (odds_collector.py) e não estava sendo checada; TELEGRAM_* faltando
    # significa degradação silenciosa sem alerta nenhum.
    vars_to_check = [
        "POLY_PRIVATE_KEY",
        "POLY_API_KEY",
        "POLY_API_SECRET",
        "POLY_API_PASSPHRASE",
        "ODDS_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "TRADING_MODE",
    ]
    print("\n  Variáveis de ambiente:")
    for var in vars_to_check:
        val = os.getenv(var, "")
        if val and val not in ("...", "seu_api_key", "0x_sua_private_key_aqui"):
            print(f"    {var}: configurada")
        else:
            print(f"    {var}: NAO configurada")


if __name__ == "__main__":
    print("=== Validando conexoes com APIs do Polymarket ===")
    check_env_vars()

    print("\n  Testando APIs:")
    # P2-42: TheGraph é fonte secundária e não-bloqueante de propósito — fica
    # fora do gate de sucesso (critical), mas o resultado real é reportado,
    # não escondido atrás de um True fixo.
    critical = [
        test_gamma_api(),
        test_clob_api_public(),
        test_clob_api_auth(),
    ]
    thegraph_ok = test_thegraph()
    if not thegraph_ok:
        print("  [AVISO] TheGraph indisponível — não bloqueia (fonte secundária).")

    print()
    if all(critical):
        print("=== Todas as conexoes validadas com sucesso! ===")
        print("\nProximos passos:")
        print("  1. Configure o .env com suas credenciais Polymarket")
        print("  2. Execute: python pipeline/fetch_markets.py")
        sys.exit(0)
    else:
        print("=== Algumas conexoes falharam. Verifique os erros acima. ===")
        sys.exit(1)
