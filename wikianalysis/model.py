"""Pipeline di classificazione Spark ML e sua valutazione.

Il classificatore predice la ``categoria`` di un articolo dal testo
(``summary`` + ``documents``). La pipeline è volutamente semplice — una
baseline interpretabile eseguibile in locale, non un modello ottimizzato.

Sequenza degli stage: indicizzazione dell'etichetta → tokenizzazione →
rimozione stop word → ``CountVectorizer`` → ``StandardScaler`` → regressione
logistica multiclasse.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import (
    CountVectorizer,
    CountVectorizerModel,
    StandardScaler,
    StopWordsRemover,
    StringIndexer,
    StringIndexerModel,
    Tokenizer,
)
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import DataFrame
from pyspark.sql.functions import concat_ws

#: Dimensione massima del vocabolario del ``CountVectorizer``.
DEFAULT_VOCAB_SIZE = 10000

#: Colonna di etichetta in ingresso.
LABEL_COLUMN = "categoria"

#: Nome di una metrica accettato da ``MulticlassClassificationEvaluator``.
MetricName = Literal["accuracy", "weightedPrecision", "weightedRecall", "f1"]

#: Metriche calcolate da :func:`evaluate`, mappate al nome Spark.
METRIC_NAMES: dict[str, MetricName] = {
    "accuracy": "accuracy",
    "weighted_precision": "weightedPrecision",
    "weighted_recall": "weightedRecall",
    "f1": "f1",
}


def build_features(df: DataFrame) -> DataFrame:
    """Concatena ``summary`` e ``documents`` in un'unica colonna ``content``.

    Args:
        df: DataFrame pulito con le colonne testuali e l'etichetta.

    Returns:
        DataFrame con la sola etichetta e la colonna ``content``.
    """
    return df.withColumn("content", concat_ws(" ", df["summary"], df["documents"])).drop(
        "title", "summary", "documents"
    )


def build_pipeline(
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    label_col: str = LABEL_COLUMN,
) -> Pipeline:
    """Costruisce la pipeline di classificazione.

    Lo ``StopWordsRemover`` produce la colonna ``filtered`` che alimenta il
    ``CountVectorizer``: senza questo collegamento le stop word non verrebbero
    rimosse prima della vettorizzazione.

    Args:
        vocab_size: dimensione massima del vocabolario.
        label_col: colonna di etichetta da indicizzare.

    Returns:
        La ``Pipeline`` non addestrata.
    """
    label_indexer = StringIndexer(inputCol=label_col, outputCol="label")
    tokenizer = Tokenizer(inputCol="content", outputCol="tokens")
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
    vectorizer = CountVectorizer(inputCol="filtered", outputCol="counts", vocabSize=vocab_size)
    scaler = StandardScaler(inputCol="counts", outputCol="scaled_counts")
    lr = LogisticRegression(featuresCol="scaled_counts", labelCol="label")
    return Pipeline(stages=[label_indexer, tokenizer, remover, vectorizer, scaler, lr])


def evaluate(predictions: DataFrame) -> dict[str, float]:
    """Calcola accuracy, precision/recall pesate e F1 sulle predizioni.

    Args:
        predictions: output di ``model.transform(test_data)`` (colonne
            ``label`` e ``prediction``).

    Returns:
        Dizionario ``{nome_metrica: valore}``.
    """
    return {
        name: MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName=spark_name
        ).evaluate(predictions)
        for name, spark_name in METRIC_NAMES.items()
    }


def confusion_matrix(predictions: DataFrame) -> np.ndarray:
    """Restituisce la matrice di confusione come array NumPy.

    Args:
        predictions: output di ``model.transform(test_data)``.

    Returns:
        Matrice quadrata ``(n_classi, n_classi)`` con i conteggi.
    """
    prediction_and_labels = predictions.select("prediction", "label").rdd.map(tuple)
    return MulticlassMetrics(prediction_and_labels).confusionMatrix().toArray()


def label_names(model: PipelineModel) -> list[str]:
    """Estrae i nomi delle categorie nell'ordine degli indici di etichetta.

    Lo ``StringIndexerModel`` è il primo stage della pipeline addestrata: il suo
    attributo ``labels`` mappa l'indice numerico al nome originale.

    Args:
        model: pipeline addestrata.

    Returns:
        I nomi delle categorie ordinati per indice.
    """
    indexer = cast(StringIndexerModel, model.stages[0])
    return list(indexer.labels)


def top_tokens(model: PipelineModel, *, n: int = 20) -> pd.DataFrame:
    """Restituisce i ``n`` token più importanti per il modello.

    L'importanza è la media del valore assoluto dei coefficienti della
    regressione logistica sulle classi.

    Args:
        model: pipeline addestrata.
        n: numero di token da restituire.

    Returns:
        DataFrame Pandas con colonne ``token`` e ``peso``, ordinato per peso.
    """
    vectorizer = cast(CountVectorizerModel, model.stages[3])
    lr = cast(LogisticRegressionModel, model.stages[-1])

    vocab = vectorizer.vocabulary
    coefs = lr.coefficientMatrix.toArray()  # (n_classi, vocab_size)

    importance = np.abs(coefs).mean(axis=0)
    top_idx = importance.argsort()[-n:][::-1]
    return pd.DataFrame(
        {
            "token": [vocab[i] for i in top_idx],
            "peso": importance[top_idx],
        }
    )
