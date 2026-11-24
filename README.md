# Internet Detective: Fake Review Classifier

A small, educational text classifier (TF-IDF + Logistic Regression) with a Streamlit app.

## Read this first: what the model can and cannot do

- It learns to separate **"CG" (computer-generated) from "OR" (original) reviews in one
  specific dataset**. On that dataset's held-out test split it scores
  **95.0% accuracy** (5-fold CV macro F1 94.3%).
- That is **not** a general fake-review detector. On the separate Amazon file it scores
  **50.0% accuracy with 3.6% Fake recall**: chance level,
  and it calls almost everything Genuine.
- Tuning does not fix that. A text-only model trained *on* Amazon reaches only
  66.8%. The `VERIFIED_PURCHASE` column alone reaches
  78.3%, but that is a dataset shortcut, not evidence of fakery.
- **Model confidence does not detect this failure.** On Amazon the model is confidently
  wrong (about 94% of reviews clear the default 60% bar, with ~50% accuracy). The app's
  "Unsure" result only catches short text, non-English text and low-confidence cases.
- CG text comes from an older text generator. Do not expect it to catch modern LLM text or
  human-written fake reviews.

## Data

1. `data/fake_reviews.csv`: 40,432 rows (`category`, `rating`, `label`, `text_`). `CG` -> Fake,
   `OR` -> Genuine. The only training source.
2. `data/amazon_reviews.txt`: 21,000 tab-separated rows. `__label1__` -> Fake, `__label2__` ->
   Genuine. Used only for external evaluation, never for fitting or model selection.

## Files

- `app.py`: Streamlit interface (single review, explanations, Unsure handling, CSV batch
  prediction, correction export, dataset and evaluation views).
- `review_logic.py`: prediction, quality-check, explanation and batch helpers (no Streamlit,
  unit-tested).
- `train_model.py`: stratified 80/20 split, 5-fold CV on training rows to choose `C`
  (grid {1, 4, 10, 30}), final fit, held-out and Amazon evaluation, reference baselines.
- `compare_models.py`: optional CV-only comparison (LR at several C, Naive Bayes, Linear SVM).
- `utils.py`: paths, label mapping, text cleaning, CSV-injection sanitizing.
- `tests/test_core.py`: fast unit tests (`pip install -r requirements-dev.txt && pytest`).
- `models/`: `model.joblib`, `vectorizer.joblib`, `metrics.json`.

## Setup

Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The shipped model files are pickles and load reliably only with the pinned
`scikit-learn==1.9.1`. If your version differs, the app shows a **Retrain model now**
button (about a minute), or run `python3 train_model.py` yourself.

## Model

TF-IDF: up to 50,000 unigram+bigram features, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`,
stop words kept so negations survive. Logistic Regression with `C=10`,
chosen by CV on training rows only; the test split is never used for selection. The vectorizer
is fit on training rows only. Text cleaning: Unicode normalization, HTML/URL handling,
whitespace. Rating, category and other metadata are not used.

## App behaviour

- **Unsure** is shown for fewer than 5 words, fewer than 60% of words in the model
  vocabulary (English reviews sit above 85%; Spanish/French/German samples around 20-27%), or
  confidence below the sidebar threshold (0.55-0.95).
- **Explanations** highlight words with the largest positive contribution to the predicted
  class. They are associations, not proof.
- **CSV batch** adds model label, app result, confidence and status. Exported CSVs prefix cells
  that start with `=`, `+`, `-` or `@` with `'` to prevent spreadsheet formula injection.
- **Corrections** are queued in the session and exportable. They do not retrain the model.

## Changes from the first version

Fixed: ineffective confidence slider range, weak non-English detection, dead `n't` negation,
one-letter words ignored in the word count, CSV formula injection, duplicate corrections,
unpinned dependencies, misleading `best_*` names. Added: CV-tuned `C` (CV macro F1 92.8% -> 94.3%),
Amazon reference baselines, retrain button, unit tests, `.gitignore`, and honest limitation
text in the app and README.
