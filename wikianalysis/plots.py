"""Grafici del progetto: word cloud per categoria e confusion matrix.

Le funzioni ricevono dati già aggregati (piccoli, lato Pandas/NumPy) e
restituiscono una ``Figure`` matplotlib, così il notebook resta una sottile
orchestrazione e la logica di disegno è testabile a parte.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from wordcloud import WordCloud

#: Colonna di categoria nei DataFrame di input.
CATEGORY_COLUMN = "categoria"


def plot_wordclouds(
    freqs_by_category: pd.DataFrame,
    *,
    category_col: str = CATEGORY_COLUMN,
    freqs_col: str = "freqs",
    n_cols: int = 5,
) -> Figure:
    """Disegna una griglia di word cloud, una per categoria.

    Args:
        freqs_by_category: DataFrame Pandas con una colonna di categoria e una
            colonna di mappe ``{token: conteggio}`` (output di
            :func:`wikianalysis.eda.top_tokens_by_category` portato in Pandas).
        category_col: nome della colonna di categoria.
        freqs_col: nome della colonna con le frequenze.
        n_cols: numero di colonne della griglia.

    Returns:
        La ``Figure`` con la griglia di word cloud.
    """
    data = freqs_by_category.sort_values(category_col).reset_index(drop=True)
    n_cat = len(data)
    n_rows = math.ceil(n_cat / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6))
    axes = np.atleast_1d(axes).flatten()

    for idx, (_, row) in enumerate(data.iterrows()):
        ax = axes[idx]
        freqs = {token: int(cnt) for token, cnt in row[freqs_col].items()}
        wc = WordCloud(
            width=400,
            height=200,
            background_color="white",
            colormap="tab10",
            prefer_horizontal=1.0,
        ).generate_from_frequencies(freqs)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(row[category_col], fontsize=20, pad=10)
        ax.axis("off")

    for ax in axes[n_cat:]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: list[str]) -> Figure:
    """Disegna la matrice di confusione come heatmap annotata.

    Args:
        cm: matrice di confusione ``(n_classi, n_classi)``.
        labels: nomi delle categorie nell'ordine degli indici.

    Returns:
        La ``Figure`` con la heatmap.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predetto")
    ax.set_ylabel("Reale")
    ax.set_title("Confusion Matrix")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    return fig
