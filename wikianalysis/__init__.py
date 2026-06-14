"""Analisi e classificazione degli articoli di Wikipedia con Apache Spark.

Il package raccoglie la logica ingegnerizzata del progetto (caricamento dati,
pulizia, EDA e classificazione) in moduli riutilizzabili e testabili, separati
dal notebook narrativo.
"""

from wikianalysis import data, eda, model, plots
from wikianalysis.data import build_sample, download_dataset, read_raw, scrub
from wikianalysis.spark import get_spark_session

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "build_sample",
    "data",
    "download_dataset",
    "eda",
    "get_spark_session",
    "model",
    "plots",
    "read_raw",
    "scrub",
]
