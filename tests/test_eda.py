"""Test del modulo :mod:`wikianalysis.eda` (conteggi, lunghezze, token)."""

from __future__ import annotations

from pyspark.sql import SparkSession

from wikianalysis import eda

COLS = ["title", "summary", "documents", "categoria"]


def test_count_by_category_counts_and_sorts(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        [
            ("t", "s", "d", "a"),
            ("t", "s", "d", "a"),
            ("t", "s", "d", "b"),
        ],
        COLS,
    )
    rows = eda.count_by_category(df).collect()
    counts = {r["categoria"]: r["documents_cnt"] for r in rows}
    assert counts == {"a": 2, "b": 1}
    # Ordinamento decrescente: la categoria più frequente è prima.
    assert rows[0]["categoria"] == "a"


def test_length_stats_by_category(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        [
            ("t", "s", "ab", "a"),  # len 2
            ("t", "s", "abcd", "a"),  # len 4
        ],
        COLS,
    )
    row = eda.length_stats_by_category(df).collect()[0]
    assert row["max_documents_len"] == 4
    assert row["min_documents_len"] == 2
    assert row["avg_documents_len"] == 3.0


def test_build_stopwords_includes_expected_terms() -> None:
    sw = eda.build_stopwords()
    assert "the" in sw  # stop word inglese di base
    assert "one" in sw  # numero scritto
    assert "http" in sw  # frammento custom
    assert len(sw) > len(eda.NUMBER_WORDS)


def test_build_stopwords_extra_are_appended() -> None:
    sw = eda.build_stopwords(extra=["zzz_custom"])
    assert "zzz_custom" in sw


def test_top_tokens_by_category_removes_stopwords_and_counts(spark: SparkSession) -> None:
    # "the" è stop word e va rimossa; "spark" ricorre 2 volte -> entra (min_df=2).
    df = spark.createDataFrame(
        [
            ("t", "s", "the spark spark engine", "tech"),
            ("t", "s", "the spark cluster", "tech"),
        ],
        COLS,
    )
    rows = eda.top_tokens_by_category(df, top_k=10, min_df=2).collect()
    freqs = rows[0]["freqs"]
    assert "the" not in freqs
    assert freqs["spark"] == 3


def test_top_tokens_respects_min_df(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        [("t", "s", "alpha beta beta", "c")],
        COLS,
    )
    freqs = eda.top_tokens_by_category(df, top_k=10, min_df=2).collect()[0]["freqs"]
    assert "beta" in freqs  # ricorre 2 volte
    assert "alpha" not in freqs  # ricorre 1 volta, sotto min_df
