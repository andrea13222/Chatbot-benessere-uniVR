import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "sito_scraped"

URL_INIZIALE = "https://www.univr.it/it/benessere"
MAX_PAGINE = 30
PAUSA_SECONDI = 1.5
SOLO_STESSO_DOMINIO = True

HEADERS = {
    "User-Agent": "Mozilla/5.0 (scraping sito per tesi universitaria)"
}


def _dominio(url):
    return urlparse(url).netloc


def _puo_fare_scraping(url):
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(HEADERS["User-Agent"], url)
    except Exception:
        return True


def _estrai_testo_e_link(html, url_pagina):
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    testo = soup.get_text(separator="\n")
    testo_pulito = "\n".join(riga.strip() for riga in testo.split("\n") if riga.strip())

    link_trovati = []
    for a in soup.find_all("a", href=True):
        link_assoluto = urljoin(url_pagina, a["href"])
        link_assoluto = link_assoluto.split("#")[0]
        if link_assoluto.startswith("http"):
            link_trovati.append(link_assoluto)

    return testo_pulito, link_trovati


def _nome_file_da_url(url):
    parsed = urlparse(url)
    percorso = parsed.path.strip("/").replace("/", "_") or "home"
    return f"{percorso}.txt"


def scrape_sito(url_iniziale, max_pagine=MAX_PAGINE):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dominio_base = _dominio(url_iniziale)
    da_visitare = [url_iniziale]
    visitate = set()
    pagine_salvate = 0

    while da_visitare and pagine_salvate < max_pagine:
        url = da_visitare.pop(0)

        if url in visitate:
            continue
        visitate.add(url)

        if SOLO_STESSO_DOMINIO and _dominio(url) != dominio_base:
            continue

        if not _puo_fare_scraping(url):
            print(f"  [saltata, robots.txt non permette] {url}")
            continue

        try:
            print(f"[{pagine_salvate + 1}/{max_pagine}] Scarico: {url}")
            risposta = requests.get(url, headers=HEADERS, timeout=10)
            risposta.raise_for_status()
        except requests.RequestException as e:
            print(f"  Errore {url}: {e}")
            continue

        testo, link_trovati = _estrai_testo_e_link(risposta.text, url)

        if len(testo) < 100:
            print(f"  Pagina non rlevante, skip")
        else:
            nome_file = _nome_file_da_url(url)
            percorso_file = OUTPUT_DIR / nome_file
            with open(percorso_file, "w", encoding="utf-8") as f:
                f.write(f"URL originale: {url}\n\n{testo}")
            pagine_salvate += 1
            print(f"  Salvato in: {percorso_file}")

        for link in link_trovati:
            if link not in visitate and link not in da_visitare:
                da_visitare.append(link)

        time.sleep(PAUSA_SECONDI)

    print(f"\nCompletato: {pagine_salvate} pagine salvate in {OUTPUT_DIR}")


if __name__ == "__main__":
    print(f"Inizio scraping da: {URL_INIZIALE}")
    scrape_sito(URL_INIZIALE)
