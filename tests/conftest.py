"""Fixture condivise per la suite di test.

La ``SparkSession`` è **session-scoped**: avviare la JVM è costoso, quindi la si
crea una sola volta per l'intera sessione di test. Il backend matplotlib è
forzato a ``Agg`` (non interattivo) così i test sui grafici non aprono finestre.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import matplotlib
import pytest
from pyspark.sql import DataFrame, SparkSession

from wikianalysis import data, get_spark_session

matplotlib.use("Agg")


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    """SparkSession locale a basso parallelismo per i test."""
    session = get_spark_session("wikianalysis-tests", shuffle_partitions=2)
    yield session
    session.stop()


@pytest.fixture(scope="session")
def sample_clean(spark: SparkSession) -> DataFrame:
    """Sample versionato letto e pulito, riusato dai test di integrazione."""
    raw = data.read_raw(spark, data.SAMPLE_CSV)
    return data.scrub(raw).cache()


@pytest.fixture(scope="session")
def sample_path() -> Path:
    """Percorso del campione versionato."""
    return data.SAMPLE_CSV
