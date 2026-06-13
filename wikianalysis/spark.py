"""Creazione di una ``SparkSession`` locale, riproducibile e a basso rumore.

Centralizzare qui la configurazione evita di duplicarla tra notebook, test e CI,
e risolve un paio di insidie tipiche dell'esecuzione locale (soprattutto su
Windows): il worker Python deve usare lo stesso interprete del driver, altrimenti
Spark fallisce con un crash opaco del worker.
"""

from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession


def get_spark_session(
    app_name: str = "wikipedia-analysis",
    master: str = "local[*]",
    *,
    shuffle_partitions: int = 8,
) -> SparkSession:
    """Crea (o riusa) una ``SparkSession`` locale pronta per l'analisi.

    Args:
        app_name: nome dell'applicazione mostrato nella Spark UI / nei log.
        master: URL del master Spark. Default ``local[*]`` (tutti i core locali).
        shuffle_partitions: numero di partizioni di shuffle. Il default di Spark
            (200) è sovradimensionato per un dataset che gira su una singola
            macchina e rende le query inutilmente lente.

    Returns:
        Una ``SparkSession`` attiva. Le chiamate successive riusano la stessa
        sessione (semantica ``getOrCreate``).
    """
    # Il worker Python deve essere lo stesso interprete del driver: senza questo
    # allineamento, su Windows Spark spawna un worker incompatibile che crasha.
    python_exe = sys.executable
    os.environ.setdefault("PYSPARK_PYTHON", python_exe)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", python_exe)

    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
