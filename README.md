# VAERS Severe Reaction Prediction

Progetto universitario di Machine Learning per predire il rischio di reazione grave al vaccino COVID-19 usando dati VAERS.

## Obiettivo

Costruire e valutare un modello predittivo che, dato il profilo clinico e anamnestico del paziente, stimi la probabilita di sviluppare una reazione grave.

## Struttura Del Progetto

```text
JupyterProject/
├── data/
│   ├── raw/                  # dataset originale
│   ├── interim/              # step intermedi (split, no-scale, npl)
│   ├── processed/            # feature pronte per training (encoded/scaled/smote)
│   └── evaluation/           # output di analisi errori (false negatives, campioni TP)
├── notebooks/
│   ├── 00_exploration/
│   ├── 01_data_cleaning/
│   ├── 02_feature_engineering/
│   ├── 03_modeling/
│   └── 04_evaluation/        # notebook + launcher script compatibili
├── models/
│   ├── checkpoints/          # modelli intermedi
│   └── production/           # modello finale pronto all'uso
├── artifacts/
│   └── overleaf/             # pacchetti .zip pronti per upload su Overleaf
├── reports/
│   ├── figures/              # mirror legacy dei grafici (compatibilità)
│   └── metrics/              # report JSON/CSV degli esperimenti
├── report/
│   ├── main.tex              # report accademico in LaTeX
│   └── figures/              # cartella canonica delle figure usate da main.tex
├── src/
│   ├── data/                 # moduli data layer
│   ├── features/             # moduli feature engineering
│   ├── modeling/             # moduli modellazione
│   ├── pipelines/            # pipeline end-to-end
│   ├── evaluation/           # script python di test/benchmark
│   └── utils/                # utility python
├── requirements.txt
└── README.md
```

## Workflow Consigliato

1. `notebooks/00_exploration`: analisi preliminare.
2. `notebooks/01_data_cleaning`: pulizia, deduplicazione, target engineering, split.
3. `notebooks/02_feature_engineering`: trasformazioni e nuove feature.
4. `notebooks/03_modeling`: training, bilanciamento classi, tuning.
5. `src/evaluation`: benchmark e valutazione finale script-based.
6. `report/main.tex`: aggiornamento report finale.

## Setup Rapido

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Note

- I notebook sono stati riorganizzati per fase della pipeline.
- I dataset sono separati per livello di maturita (`raw`, `interim`, `processed`).
- I modelli `.pkl` sono centralizzati in `models/`.
- Gli script Python operativi sono in `src/evaluation`.
- In `notebooks/04_evaluation` sono presenti launcher compatibili che richiamano gli script in `src/evaluation`.

## Test Avanzato Del Modello

Per confrontare automaticamente piu strategie (SMOTE vs class weighting vs blend),
ottimizzare la soglia e salvare il modello migliore:

```bash
.venv/bin/python src/evaluation/test5_advanced_model_test.py \
  --objective precision_at_recall \
  --min-recall 0.60 \
  --save-model
```

Output principali:

- `reports/metrics/advanced_model_benchmark_summary.csv`
- `reports/metrics/advanced_model_benchmark_report.json`
- `models/production/severe_model_optimized.pkl`

## Test4 Ottimizzato

Il notebook `/Users/marcodonatiello/PycharmProjects/JupyterProject/notebooks/04_evaluation/test4.ipynb`
usa la tattica risultata migliore dai benchmark: `LightGBM` pesato e ottimizzato con
selezione soglia `best_precision_with_recall>=0.58`.

## Script Di Valutazione Recenti

- `src/evaluation/test6_stacking.py`: orthogonal stacking (LGBM + linear + KNN + MLP).
- `src/evaluation/test7_lgbm_stacking.py`: stacking con varianti LGBM (GBDT/DART/GOSS).
- `src/evaluation/test8_recall_first_policy.py`: policy recall-first con vincolo di precisione.
- `src/evaluation/test9_retrain_recall70.py`: campagna retraining per target recall 0.70.

## Generazione Grafici Per Report

Per rigenerare automaticamente i grafici finali (PR/ROC, confusion matrix, feature importance, calibrazione, trade-off soglia, distribuzioni):

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python src/evaluation/generate_report_figures.py
```

Output:

- Cartella canonica: `/Users/marcodonatiello/PycharmProjects/JupyterProject/report/figures/`
- Mirror legacy: `/Users/marcodonatiello/PycharmProjects/JupyterProject/reports/figures/`
   
