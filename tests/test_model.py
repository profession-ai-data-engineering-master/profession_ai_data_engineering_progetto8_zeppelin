"""Test del modulo :mod:`wikianalysis.model` (feature, pipeline, valutazione)."""

from __future__ import annotations

import pytest
from pyspark.ml import PipelineModel
from pyspark.ml.feature import CountVectorizer
from pyspark.sql import DataFrame, SparkSession

from wikianalysis import model

COLS = ["title", "summary", "documents", "categoria"]


@pytest.fixture(scope="module")
def split(sample_clean: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Split train/test riproducibile sul sample, pronto per la pipeline."""
    features = model.build_features(sample_clean)
    train, test = features.randomSplit([0.8, 0.2], seed=42)
    return train, test


@pytest.fixture(scope="module")
def fitted(split: tuple[DataFrame, DataFrame]) -> PipelineModel:
    """Pipeline addestrata una sola volta e riusata dai test."""
    train, _ = split
    return model.build_pipeline().fit(train)


def test_build_features_creates_content(spark: SparkSession) -> None:
    df = spark.createDataFrame([("t", "hello", "world", "cat")], COLS)
    out = model.build_features(df).collect()[0]
    assert out["content"] == "hello world"
    assert out["categoria"] == "cat"


def test_build_features_drops_text_columns(spark: SparkSession) -> None:
    df = spark.createDataFrame([("t", "s", "d", "cat")], COLS)
    cols = model.build_features(df).columns
    assert set(cols) == {"categoria", "content"}


def test_pipeline_has_six_stages() -> None:
    stages = model.build_pipeline().getStages()
    assert len(stages) == 6


def test_vectorizer_reads_filtered_column() -> None:
    """Regressione del bug del gradino 4: il CountVectorizer deve leggere
    ``filtered`` (output dello StopWordsRemover), non ``tokens``."""
    vectorizer = next(
        s for s in model.build_pipeline().getStages() if isinstance(s, CountVectorizer)
    )
    assert vectorizer.getInputCol() == "filtered"


def test_label_names_covers_all_categories(fitted: PipelineModel) -> None:
    labels = model.label_names(fitted)
    assert len(labels) == 15
    assert "medicine" in labels


def test_predictions_have_prediction_column(
    fitted: PipelineModel, split: tuple[DataFrame, DataFrame]
) -> None:
    _, test = split
    preds = fitted.transform(test)
    assert "prediction" in preds.columns
    assert preds.count() > 0


def test_evaluate_returns_metrics_in_range(
    fitted: PipelineModel, split: tuple[DataFrame, DataFrame]
) -> None:
    _, test = split
    metrics = model.evaluate(fitted.transform(test))
    assert set(metrics) == {"accuracy", "weighted_precision", "weighted_recall", "f1"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_confusion_matrix_is_square(
    fitted: PipelineModel, split: tuple[DataFrame, DataFrame]
) -> None:
    _, test = split
    cm = model.confusion_matrix(fitted.transform(test))
    assert cm.ndim == 2
    assert cm.shape[0] == cm.shape[1]


def test_top_tokens_shape_and_order(fitted: PipelineModel) -> None:
    top = model.top_tokens(fitted, n=10)
    assert list(top.columns) == ["token", "peso"]
    assert len(top) == 10
    # I pesi sono ordinati in modo decrescente.
    assert list(top["peso"]) == sorted(top["peso"], reverse=True)


def test_training_is_reproducible(split: tuple[DataFrame, DataFrame]) -> None:
    """Stesso seed e stessi dati -> stesse metriche."""
    train, test = split
    m1 = model.build_pipeline().fit(train)
    m2 = model.build_pipeline().fit(train)
    assert model.evaluate(m1.transform(test)) == model.evaluate(m2.transform(test))
