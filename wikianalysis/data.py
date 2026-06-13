"""Reperibilità e campionamento del dataset Wikipedia.

Il dataset completo (~957 MB) vive su S3 ed è troppo grande per il repo e per la
CI. Questo modulo offre due cose:

* :func:`download_dataset` — scarica il CSV completo in una cartella locale
  gitignorata, in modo riproducibile (idempotente, niente cloud a pagamento).
* :func:`build_sample` — estrae un campione bilanciato per categoria, piccolo e
  versionato, usato da test e CI.

Esecuzione da riga di comando::

    python -m wikianalysis.data download    # scarica il full dataset
    python -m wikianalysis.data sample      # rigenera il sample versionato
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

#: URL pubblico del dataset completo su S3.
DATASET_URL = "https://proai-datasets.s3.eu-west-3.amazonaws.com/wikipedia.csv"

#: Cartella dati, relativa alla radice del repo (``<repo>/data``).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

#: Percorso del dataset completo (gitignorato).
FULL_CSV = DATA_DIR / "wikipedia.csv"

#: Percorso del campione versionato (committato nel repo).
SAMPLE_CSV = DATA_DIR / "wikipedia_sample.csv"

#: Colonna che contiene l'etichetta di categoria.
CATEGORY_COLUMN = "categoria"

#: Numero di righe per categoria nel campione.
DEFAULT_ROWS_PER_CATEGORY = 40

#: Seed per rendere il campionamento riproducibile.
DEFAULT_SEED = 42


def download_dataset(
    dest: Path = FULL_CSV,
    *,
    url: str = DATASET_URL,
    force: bool = False,
) -> Path:
    """Scarica il dataset completo da S3 in ``dest``.

    È idempotente: se il file esiste già non lo riscarica, a meno che
    ``force=True``. Il download usa solo la standard library (nessuna dipendenza
    aggiuntiva) e mostra l'avanzamento su stderr.

    Args:
        dest: percorso di destinazione del CSV completo.
        url: URL sorgente del dataset.
        force: se ``True`` riscarica anche se il file è già presente.

    Returns:
        Il percorso del file scaricato.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"Dataset già presente in {dest} ({dest.stat().st_size:,} byte). Skip.")
        return dest

    print(f"Download da {url} ...", file=sys.stderr)

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100.0, downloaded * 100.0 / total_size)
        print(f"\r  {pct:5.1f}%  ({downloaded:,}/{total_size:,} byte)", end="", file=sys.stderr)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)  # noqa: S310
    print(file=sys.stderr)
    print(f"Salvato in {dest} ({dest.stat().st_size:,} byte).")
    return dest


def build_sample(
    rows_per_category: int = DEFAULT_ROWS_PER_CATEGORY,
    *,
    src: Path = FULL_CSV,
    dest: Path = SAMPLE_CSV,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Crea un campione bilanciato per categoria a partire dal dataset completo.

    Estrae fino a ``rows_per_category`` righe casuali per ogni categoria, così il
    sample resta piccolo ma rappresentativo di tutte le classi. Il risultato è
    deterministico a parità di ``seed``.

    Args:
        rows_per_category: righe da campionare per ciascuna categoria.
        src: dataset completo da cui campionare (deve esistere).
        dest: percorso del campione da scrivere.
        seed: seed del campionamento.

    Returns:
        Il percorso del campione scritto.

    Raises:
        FileNotFoundError: se ``src`` non esiste (scaricare prima il dataset).
    """
    if not src.exists():
        raise FileNotFoundError(
            f"Dataset completo non trovato in {src}. "
            "Eseguire prima 'python -m wikianalysis.data download'."
        )

    df = pd.read_csv(src, index_col=0)

    def _take(group: pd.DataFrame) -> pd.DataFrame:
        n = min(rows_per_category, len(group))
        return group.sample(n=n, random_state=seed)

    sample = (
        df.groupby(CATEGORY_COLUMN, group_keys=False)[df.columns]
        .apply(_take)
        .sort_values(CATEGORY_COLUMN)
        .reset_index(drop=True)
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(dest, index=False)
    counts = sample[CATEGORY_COLUMN].value_counts().sort_index()
    print(f"Sample scritto in {dest}: {len(sample)} righe.")
    print("Righe per categoria:")
    for category, count in counts.items():
        print(f"  {category}: {count}")
    return dest


def main(argv: list[str] | None = None) -> None:
    """Entry point CLI per scaricare il dataset o rigenerare il sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser("download", help="scarica il dataset completo da S3")
    p_download.add_argument("--force", action="store_true", help="riscarica anche se presente")

    p_sample = sub.add_parser("sample", help="rigenera il campione versionato")
    p_sample.add_argument(
        "-n",
        "--rows-per-category",
        type=int,
        default=DEFAULT_ROWS_PER_CATEGORY,
        help=f"righe per categoria (default: {DEFAULT_ROWS_PER_CATEGORY})",
    )

    args = parser.parse_args(argv)
    if args.command == "download":
        download_dataset(force=args.force)
    elif args.command == "sample":
        build_sample(rows_per_category=args.rows_per_category)


if __name__ == "__main__":
    main()
