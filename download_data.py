"""
Descarga el dataset Bank Marketing (UCI) en data/bank-full.csv.

El dataset ya está incluido en el repositorio, así que este script solo es
necesario si borraste el archivo o quieres volver a obtenerlo desde la fuente
original.

Fuente:
    https://archive.ics.uci.edu/dataset/222/bank+marketing

Uso:
    python download_data.py
"""

from __future__ import annotations

import io
import ssl
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "bank-full.csv"


def fetch(url: str) -> bytes:
    """Descarga una URL; reintenta sin verificar el certificado si este falla."""
    try:
        with urllib.request.urlopen(url, timeout=90) as response:
            return response.read()
    except (ssl.SSLError, urllib.error.URLError):
        # El certificado del sitio de la UCI suele estar vencido.
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, timeout=90, context=context) as response:
            return response.read()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {URL} ...")
    outer = zipfile.ZipFile(io.BytesIO(fetch(URL)))
    inner = zipfile.ZipFile(io.BytesIO(outer.read("bank.zip")))
    DATA_PATH.write_bytes(inner.read("bank-full.csv"))
    size_mb = DATA_PATH.stat().st_size / 1_000_000
    print(f"Se guardó {DATA_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
