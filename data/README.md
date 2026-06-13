# Dati

## Dataset completo (non versionato)

Il dataset completo degli articoli di Wikipedia vive su S3 ed è **gitignorato**
(troppo grande per il repo e per la CI):

| Proprietà | Valore |
|-----------|--------|
| URL | <https://proai-datasets.s3.eu-west-3.amazonaws.com/wikipedia.csv> |
| Dimensione | ~957 MB (1.003.477.941 byte) |
| Righe | 153.232 |
| Colonne | `title`, `summary`, `documents`, `categoria` (+ indice) |
| Categorie | 15 (≈10.000 articoli ciascuna) |

Le categorie: `culture`, `economics`, `energy`, `engineering`, `finance`,
`humanities`, `medicine`, `pets`, `politics`, `research`, `science`, `sports`,
`technology`, `trade`, `transport`.

### Scaricarlo

```bash
python -m wikianalysis.data download
```

Il file viene salvato in `data/wikipedia.csv` (idempotente: se è già presente
non viene riscaricato; usa `--force` per forzare).

## Campione versionato (`wikipedia_sample.csv`)

Per test e CI è committato un **campione bilanciato**: 40 articoli casuali per
categoria (600 righe, ~4 MB), riproducibile con seed fisso. Permette di eseguire
EDA, pipeline e test senza scaricare ~1 GB.

### Rigenerarlo (richiede il dataset completo)

```bash
python -m wikianalysis.data sample          # 40 righe/categoria (default)
python -m wikianalysis.data sample -n 20    # campione ancora più piccolo
```
