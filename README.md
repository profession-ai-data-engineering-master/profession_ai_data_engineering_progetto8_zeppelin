# Analisi e Classificazione di Wikipedia con Apache Spark

[![CI](https://github.com/profession-ai-data-engineering-master/profession_ai_data_engineering_progetto8/actions/workflows/ci.yml/badge.svg)](https://github.com/profession-ai-data-engineering-master/profession_ai_data_engineering_progetto8/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Spark](https://img.shields.io/badge/Apache%20Spark-4.1-orange)
![coverage](https://img.shields.io/badge/coverage-87%25-brightgreen)

Analisi esplorativa e **classificazione automatica** di ~153.000 articoli di Wikipedia in 15 categorie tematiche, con **Apache Spark**. Caso d'uso fittizio *Wikidata Insights*: capire la distribuzione dei contenuti e fornire un classificatore che categorizzi i nuovi articoli a partire dal testo.

> Progetto del Master in Data Engineering di ProfessionAI — corso *Big Data con Apache Spark*.

## Dal cloud al locale

L'esercizio era nato su **Zeppelin/Databricks**, vincolato dai limiti del piano gratuito (download `wget` capato a 500 MB, librerie `spark.ml` non disponibili). Questo repo lo **porta a Spark in locale**: una `SparkSession` esplicita, niente magic di Zeppelin (`%sh`, `%sql`) né dipendenze cloud a pagamento. Il notebook **gira gratis su qualsiasi macchina** — semplicemente più lento di un cluster — così chiunque (recruiter inclusi) può riprodurlo senza spendere nulla.

## Dataset

| | Dataset completo | Campione versionato |
|---|---|---|
| File | `data/wikipedia.csv` (gitignorato) | `data/wikipedia_sample.csv` (committato) |
| Origine | [S3](https://proai-datasets.s3.eu-west-3.amazonaws.com/wikipedia.csv) (~957 MB) | 40 articoli/categoria, seed fisso |
| Righe | 153.232 | 600 (~4 MB) |
| Uso | notebook / analisi completa | test e CI |

Colonne: `title`, `summary`, `documents`, `categoria`. Le 15 categorie: `culture`, `economics`, `energy`, `engineering`, `finance`, `humanities`, `medicine`, `pets`, `politics`, `research`, `science`, `sports`, `technology`, `trade`, `transport`.

```bash
python -m wikianalysis.data download   # scarica il full (idempotente)
python -m wikianalysis.data sample      # rigenera il campione versionato
```

## Approccio

Il notebook segue il framework **OSEMN**, con tutta la logica incapsulata nel package `wikianalysis` e una pipeline **Spark ML**:

1. **Obtain** — download riproducibile del CSV e lettura in uno Spark DataFrame.
2. **Scrub** — de-duplicazione, scarto dei null, pulizia del testo (rimozione markup HTML e caratteri speciali, lowercasing).
3. **Explore** — conteggi e statistiche di lunghezza per categoria, **word cloud** dei token più frequenti (con rimozione delle stop word).
4. **Model** — classificatore multiclasse: `StringIndexer` → `Tokenizer` → `StopWordsRemover` → `CountVectorizer` → `StandardScaler` → `LogisticRegression`.
5. **iNterpret** — metriche, confusion matrix, token più influenti e raccomandazioni editoriali.

## Risultati

**EDA.** Dopo la pulizia il dataset si dimezza (153.232 → **75.523** righe: metà sono duplicati) e si rivela **fortemente sbilanciato** — `medicine` domina (8.311 articoli), `politics` è marginale (243). I testi più lunghi sono in `politics` (~12.900 caratteri medi), i più corti in `pets` (~2.500).

**Classificatore** (dataset completo, split 80/20, seed 42):

| Metrica | Valore |
|---|---|
| Accuracy | 0.8426 |
| Weighted Precision | 0.8434 |
| Weighted Recall | 0.8426 |
| Weighted F1 | 0.8429 |

Un modello **semplice ma solido**: ~0.84 di accuracy come baseline interpretabile. La confusion matrix mostra confusioni solo tra categorie semanticamente affini (es. `medicine`↔`research`); i token più influenti (*hospital*, *species*, *power*, *station*…) sono coerenti con le rispettive categorie.

> Il modello è volutamente una baseline: l'obiettivo è una pipeline corretta e riproducibile in locale, non l'ottimizzazione delle prestazioni.

## Struttura della repo

```
.
├── wikianalysis/             # package: tutta la logica testabile
│   ├── spark.py              # SparkSession locale riproducibile
│   ├── data.py               # download/sample + lettura e pulizia (scrub)
│   ├── eda.py                # conteggi, lunghezze, token per word cloud
│   ├── model.py              # pipeline Spark ML, valutazione, importanza token
│   └── plots.py              # word cloud e confusion matrix
├── tests/                    # 30 test · coverage 87% (SparkSession locale)
├── profession_ai_data_engineering_progetto8.ipynb   # notebook narrativo (OSEMN)
├── pyproject.toml            # metadata, dipendenze, config ruff/mypy/pytest
└── .github/workflows/ci.yml  # CI: Java + Spark · ruff + mypy + pytest
```

Il notebook è la **narrazione**; la logica vive nel package `wikianalysis`, così è testabile, type-checkata e riutilizzabile.

## Riproducibilità

Richiede **Python 3.12** e un **JDK 17+** (Java 21 consigliato, per la JVM di Spark).

```bash
# ambiente virtuale
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# installa il package + dipendenze
pip install -e ".[dev,notebook]"

# qualità: lint, type check, test con coverage (girano sul campione)
ruff check .
mypy
pytest

# notebook (scaricare prima il full con: python -m wikianalysis.data download)
jupyter lab
```

> **Windows.** Spark in local mode richiede `winutils.exe` per Hadoop: scaricarlo, metterlo in `C:\hadoop\bin` e impostare `HADOOP_HOME=C:\hadoop`. Su Linux/macOS non serve.

## Stack

Python · Apache Spark / PySpark 4.1 · Spark ML · pandas · NumPy · Matplotlib · seaborn · wordcloud · pytest · ruff · mypy · GitHub Actions

---

<details>
<summary><strong>Consegna originale del progetto</strong> (traccia del corso)</summary>

## Descrizione del Progetto

Wikidata Insights, un'azienda leader nella gestione di contenuti digitali, è stata incaricata da Wikipedia per ottimizzare l'analisi e la categorizzazione dei contenuti di Wikipedia.
Per supportare la loro continua espansione e migliorare l'organizzazione delle informazioni, Wikidata Insights ha deciso di condurre un progetto avanzato di data analysis e machine learning.
L'obiettivo principale è comprendere meglio il vasto patrimonio di contenuti informativi offerti da Wikipedia e sviluppare un sistema di classificazione automatica che consenta di categorizzare efficacemente i nuovi articoli futuri.

## Obiettivi

### 1. Analisi Descrittiva dei Contenuti

Il primo obiettivo del progetto è condurre un'analisi esplorativa dei dati (EDA) per capire le caratteristiche dei contenuti di Wikipedia suddivisi in diverse categorie tematiche, come ad esempio: Cultura, Economia, Medicina, Tecnologia, Politica, Scienza e altre.

L'analisi esplorativa prevede:

- il conteggio degli articoli presenti per ogni categoria;
- il numero medio di parole per articolo;
- la lunghezza dell'articolo più lungo e di quello più corto per ciascuna categoria;
- la creazione di nuvole di parole rappresentative per ogni categoria, per identificare i termini più frequenti e rilevanti.

### 2. Sviluppo di un Classificatore Automatico

Il secondo obiettivo è creare un modello di machine learning capace di classificare automaticamente gli articoli in base alla loro categoria, addestrato sui dati di testo delle colonne **Sommario** (`summary`) e **Testo Completo** (`documents`).

### 3. Identificazione di Nuovi Insights

L'analisi consentirà anche di ottenere preziosi insights sui contenuti di Wikipedia, come la densità di articoli per categoria o le tendenze linguistiche associate a determinati argomenti. Queste informazioni possono aiutare Wikimedia a migliorare l'organizzazione delle pagine e a ottimizzare i propri sforzi editoriali.

## Workflow del Progetto

### Caricamento dei Dati

I dati sono salvati su S3 e reperibili al seguente link:
<https://proai-datasets.s3.eu-west-3.amazonaws.com/wikipedia.csv>

Utilizzando un framework distribuito come Databricks, i dati vengono processati in modo efficiente, partendo da un Pandas DataFrame per essere successivamente convertiti in uno Spark DataFrame e salvati come una tabella chiamata `Wikipedia`.

Per caricare il dataset e trasformarlo in una table basta eseguire su Notebook Databricks le seguenti righe di codice:

```python
!wget https://proai-datasets.s3.eu-west-3.amazonaws.com/wikipedia.csv
import pandas as pd

dataset = pd.read_csv("/databricks/driver/wikipedia.csv")
spark_df = spark.createDataFrame(dataset)
spark_df = spark_df.drop("Unnamed: 0")
spark_df.write.saveAsTable("wikipedia")
```

> **N.B.** Durante il loading del dataset, ci appoggiamo ad un dataframe Pandas. Questa non è una procedura comune e del tutto corretta. In questo caso ci permette di leggere correttamente (superando con poco sforzo il limite dei separatori) i dati con cui definire un DataFrame Spark e una Table Wikipedia.

## Risultati Attesi

1. **Ottimizzazione dell'organizzazione dei contenuti** — una visione chiara della distribuzione e delle caratteristiche dei contenuti, per identificare le categorie da espandere.
2. **Classificazione automatica** — automatizzare la categorizzazione dei nuovi articoli, migliorando efficienza e navigabilità.
3. **Nuovi insights strategici** — ottimizzare l'allocazione delle risorse editoriali e orientare le campagne informative.

## Conclusioni

Il progetto offre a Wikimedia un potente strumento di analisi dei dati e classificazione automatica per migliorare la gestione dei propri contenuti, ottimizzando la propria infrastruttura informativa attraverso tecniche di data science e machine learning.

</details>
