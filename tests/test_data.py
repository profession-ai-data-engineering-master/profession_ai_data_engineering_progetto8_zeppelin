"""Test del modulo :mod:`wikianalysis.data` (lettura, pulizia, sample)."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import SparkSession

from wikianalysis import data

REQUIRED = ["title", "summary", "documents", "categoria"]


def _df(spark: SparkSession, rows: list[tuple]) -> object:
    return spark.createDataFrame(rows, REQUIRED)


def _counts_by_category(df: object) -> dict[str, int]:
    return {r["categoria"]: r["count"] for r in df.groupBy("categoria").count().collect()}


def test_scrub_removes_html_and_special_chars(spark: SparkSession) -> None:
    df = _df(spark, [("t", "s", "<p>Hello, World!</p>", "cat")])
    cleaned = data.scrub(df).collect()[0]["documents"]
    assert cleaned == "hello world"


def test_scrub_lowercases(spark: SparkSession) -> None:
    df = _df(spark, [("t", "s", "MixedCase TEXT", "cat")])
    assert data.scrub(df).collect()[0]["documents"] == "mixedcase text"


def test_scrub_drops_duplicates(spark: SparkSession) -> None:
    df = _df(spark, [("t", "s", "doc", "cat"), ("t", "s", "doc", "cat")])
    assert data.scrub(df).count() == 1


def test_scrub_drops_rows_with_nulls(spark: SparkSession) -> None:
    df = _df(
        spark,
        [
            ("t", "s", "doc", "cat"),
            ("t2", None, "doc2", "cat"),  # summary nullo -> scartata
        ],
    )
    rows = data.scrub(df).collect()
    assert len(rows) == 1
    assert rows[0]["title"] == "t"


def test_scrub_drops_index_column_if_present(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        [("0", "t", "s", "doc", "cat")],
        ["_c0", *REQUIRED],
    )
    cleaned = data.scrub(df)
    assert "_c0" not in cleaned.columns


def test_read_raw_sample_has_expected_schema(spark: SparkSession) -> None:
    raw = data.read_raw(spark, data.SAMPLE_CSV)
    assert set(REQUIRED).issubset(set(raw.columns))
    assert raw.count() == 600


def test_raw_sample_is_balanced(spark: SparkSession) -> None:
    """Il sample *grezzo* è bilanciato a 40 righe per categoria
    (garanzia di ``build_sample``)."""
    raw = data.read_raw(spark, data.SAMPLE_CSV)
    counts = _counts_by_category(raw)
    assert len(counts) == 15
    assert all(c == 40 for c in counts.values())


def test_sample_clean_preserves_all_categories(sample_clean: object) -> None:
    """Dopo ``scrub`` (dropDuplicates + dropna) le categorie restano tutte 15,
    con al più 40 righe ciascuna (alcune righe possono cadere)."""
    counts = _counts_by_category(sample_clean)
    assert len(counts) == 15
    assert all(0 < c <= 40 for c in counts.values())


def test_download_dataset_skips_when_present(tmp_path: Path) -> None:
    dest = tmp_path / "wikipedia.csv"
    dest.write_text("dummy", encoding="utf-8")
    # Non deve riscaricare: il file esiste e force=False.
    result = data.download_dataset(dest=dest, url="http://invalid.invalid/x.csv")
    assert result == dest
    assert dest.read_text(encoding="utf-8") == "dummy"


def test_build_sample_balances_per_category(tmp_path: Path) -> None:
    src = tmp_path / "full.csv"
    dest = tmp_path / "sample.csv"
    # CSV completo sintetico: 2 categorie, righe sbilanciate.
    lines = ["idx,title,summary,documents,categoria"]
    for i in range(5):
        lines.append(f"{i},t{i},s{i},d{i},alpha")
    for i in range(5, 8):
        lines.append(f"{i},t{i},s{i},d{i},beta")
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")

    data.build_sample(rows_per_category=2, src=src, dest=dest, seed=1)

    import pandas as pd

    out = pd.read_csv(dest)
    assert out["categoria"].value_counts().to_dict() == {"alpha": 2, "beta": 2}


def test_build_sample_raises_when_source_missing(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        data.build_sample(src=tmp_path / "nope.csv", dest=tmp_path / "out.csv")
