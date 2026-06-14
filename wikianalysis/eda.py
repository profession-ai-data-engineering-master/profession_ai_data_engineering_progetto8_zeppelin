"""Analisi esplorativa (EDA) degli articoli, lato Spark.

Funzioni pure che prendono un DataFrame pulito (output di
:func:`wikianalysis.data.scrub`) e restituiscono aggregati pronti da
visualizzare: conteggi per categoria, statistiche di lunghezza e i token più
frequenti per categoria (base per le word cloud).
"""

from __future__ import annotations

from pyspark.ml.feature import StopWordsRemover, Tokenizer
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    asc,
    avg,
    col,
    collect_list,
    count,
    desc,
    explode,
    length,
    map_from_entries,
    row_number,
    struct,
)
from pyspark.sql.functions import (
    max as spark_max,
)
from pyspark.sql.functions import (
    min as spark_min,
)

#: Colonna che contiene l'etichetta di categoria.
CATEGORY_COLUMN = "categoria"

#: Colonna testuale su cui si basano lunghezze e word cloud.
TEXT_COLUMN = "documents"

#: Lingua per la lista di stop word di base di Spark.
DEFAULT_LANG = "english"

#: Numeri scritti in lettere: poco informativi, da trattare come stop word.
NUMBER_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
]

#: Parole ad alta frequenza ma poco discriminanti, emerse dall'esplorazione.
HIGH_DF_WORDS = ["new", "also", "first", "second", "one", "two", "three", "later"]

#: Simboli e frammenti residui da scartare nelle word cloud.
PUNCTUATION_WORDS = ["’s", "“", "”", "—", "http", "https", ""]


def count_by_category(df: DataFrame, *, category_col: str = CATEGORY_COLUMN) -> DataFrame:
    """Conta gli articoli per categoria, ordinati dal più frequente.

    Args:
        df: DataFrame pulito.
        category_col: nome della colonna di categoria.

    Returns:
        DataFrame con colonne ``categoria`` e ``documents_cnt``.
    """
    return (
        df.groupBy(category_col)
        .agg(count("*").alias("documents_cnt"))
        .sort(desc("documents_cnt"))
    )


def length_stats_by_category(
    df: DataFrame,
    *,
    category_col: str = CATEGORY_COLUMN,
    text_col: str = TEXT_COLUMN,
) -> DataFrame:
    """Calcola lunghezza massima, minima e media del testo per categoria.

    Args:
        df: DataFrame pulito.
        category_col: nome della colonna di categoria.
        text_col: colonna testuale di cui misurare la lunghezza.

    Returns:
        DataFrame con ``max/min/avg_documents_len`` per categoria, ordinato per
        lunghezza media decrescente.
    """
    return (
        df.select(category_col, length(text_col).alias("documents_len"))
        .groupBy(category_col)
        .agg(
            spark_max("documents_len").alias("max_documents_len"),
            spark_min("documents_len").alias("min_documents_len"),
            avg("documents_len").alias("avg_documents_len"),
        )
        .sort(desc("avg_documents_len"))
    )


def build_stopwords(
    lang: str = DEFAULT_LANG,
    *,
    extra: list[str] | None = None,
) -> list[str]:
    """Costruisce la lista di stop word per le word cloud.

    Unisce la lista di base di Spark per la lingua scelta con numeri scritti,
    parole ad alta frequenza poco informative e frammenti di punteggiatura.

    Args:
        lang: lingua della lista di base (passata a ``loadDefaultStopWords``).
        extra: stop word aggiuntive specifiche del caso d'uso.

    Returns:
        La lista completa di stop word.
    """
    base = StopWordsRemover.loadDefaultStopWords(lang)
    stopwords = base + NUMBER_WORDS + HIGH_DF_WORDS + PUNCTUATION_WORDS
    if extra:
        stopwords = stopwords + extra
    return stopwords


def top_tokens_by_category(
    df: DataFrame,
    *,
    top_k: int = 50,
    min_df: int = 2,
    stopwords: list[str] | None = None,
    category_col: str = CATEGORY_COLUMN,
    text_col: str = TEXT_COLUMN,
) -> DataFrame:
    """Estrae i ``top_k`` token più frequenti per categoria.

    Tokenizza il testo, rimuove le stop word, conta i token per categoria
    (scartando quelli più rari di ``min_df``) e tiene i primi ``top_k`` per
    frequenza, restituendo per ogni categoria un dizionario ``{token: conteggio}``
    pronto per :func:`wikianalysis.plots.plot_wordclouds`.

    Args:
        df: DataFrame pulito.
        top_k: numero di token da tenere per categoria.
        min_df: frequenza minima perché un token sia considerato.
        stopwords: lista di stop word; se ``None`` usa :func:`build_stopwords`.
        category_col: nome della colonna di categoria.
        text_col: colonna testuale da tokenizzare.

    Returns:
        DataFrame con ``categoria`` e ``freqs`` (mappa token→conteggio).
    """
    if stopwords is None:
        stopwords = build_stopwords()

    tokenizer = Tokenizer(inputCol=text_col, outputCol="document_tokens")
    remover = StopWordsRemover(
        inputCol="document_tokens",
        outputCol="clean_tokens",
        stopWords=stopwords,
        caseSensitive=False,
    )
    tokens = remover.transform(tokenizer.transform(df))
    tokens = tokens.select(category_col, "clean_tokens").repartition(category_col)

    counts = (
        tokens.select(category_col, explode("clean_tokens").alias("token"))
        .groupBy(category_col, "token")
        .count()
        .where(col("count") >= min_df)
    )

    ranked = Window.partitionBy(category_col).orderBy(desc("count"), asc("token"))
    return (
        counts.withColumn("rank", row_number().over(ranked))
        .where(col("rank") <= top_k)
        .groupBy(category_col)
        .agg(map_from_entries(collect_list(struct("token", "count"))).alias("freqs"))
    )
