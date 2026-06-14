"""Test del modulo :mod:`wikianalysis.plots` (word cloud, confusion matrix).

I test verificano che le funzioni producano una ``Figure`` valida senza
errori; il backend ``Agg`` (impostato in ``conftest``) evita finestre.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from wikianalysis import plots


def test_plot_confusion_matrix_returns_figure() -> None:
    cm = np.array([[5, 1], [2, 4]])
    fig = plots.plot_confusion_matrix(cm, ["a", "b"])
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_xlabel() == "Predetto"
    assert fig.axes[0].get_ylabel() == "Reale"


def test_plot_wordclouds_returns_figure() -> None:
    df = pd.DataFrame(
        {
            "categoria": ["a", "b"],
            "freqs": [{"alpha": 3, "beta": 1}, {"gamma": 2, "delta": 2}],
        }
    )
    fig = plots.plot_wordclouds(df, n_cols=2)
    assert isinstance(fig, Figure)


def test_plot_wordclouds_single_category() -> None:
    df = pd.DataFrame({"categoria": ["solo"], "freqs": [{"x": 1, "y": 2}]})
    fig = plots.plot_wordclouds(df, n_cols=5)
    assert isinstance(fig, Figure)
