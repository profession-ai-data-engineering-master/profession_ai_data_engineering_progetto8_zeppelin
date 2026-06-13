"""Analisi e classificazione degli articoli di Wikipedia con Apache Spark.

Il package raccoglie la logica ingegnerizzata del progetto (caricamento dati,
pulizia, EDA e classificazione) in moduli riutilizzabili e testabili, separati
dal notebook narrativo.
"""

from wikianalysis.data import build_sample, download_dataset
from wikianalysis.spark import get_spark_session

__version__ = "0.1.0"

__all__ = ["__version__", "build_sample", "download_dataset", "get_spark_session"]
