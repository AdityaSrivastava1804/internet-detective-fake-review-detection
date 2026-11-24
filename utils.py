"""Shared project paths, label mapping, and beginner-friendly data helpers."""

import html
import re
import unicodedata
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "fake_reviews.csv"
AMAZON_DATA_PATH = PROJECT_DIR / "data" / "amazon_reviews.txt"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Both sources are translated into the same app-facing labels.
LABEL_MAP = {
    "CG": "Fake",
    "OR": "Genuine",
    "__label1__": "Fake",
    "__label2__": "Genuine",
}
DISPLAY_LABELS = ["Fake", "Genuine"]

# Input-quality limits used by the app. English CG/OR reviews have at least ~85%
# of their words in the vocabulary (1st percentile), while Spanish/French/German
# sample reviews land at 18-27%, so 0.60 separates them cleanly.
MIN_WORDS = 5
MIN_VOCAB_COVERAGE = 0.60


def sanitize_csv_cell(value):
    """Neutralize spreadsheet formulas (CSV injection) in text cells."""
    if isinstance(value, str) and value.lstrip()[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def sanitize_csv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy safe to export as CSV and open in Excel/Sheets."""
    safe = frame.copy()
    for column in safe.columns:
        if safe[column].dtype == "object" or pd.api.types.is_string_dtype(safe[column]):
            safe[column] = safe[column].map(sanitize_csv_cell)
    return safe


def clean_review(review: object) -> str:
    """Normalize formatting without deleting sentiment or negation words."""
    text = html.unescape(str(review))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"(?is)<(script|style)\b.*?>.*?</\1>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_primary_dataset() -> pd.DataFrame:
    """Load the original CG/OR dataset in a consistent table."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Put fake_reviews.csv in the data folder."
        )

    original = pd.read_csv(DATA_PATH)
    required_original = {"text_", "label", "category", "rating"}
    missing = required_original - set(original.columns)
    if missing:
        raise ValueError(f"Original dataset is missing columns: {sorted(missing)}")

    original_rows = pd.DataFrame(
        {
            "text_": original["text_"],
            "source_label": original["label"].astype("string").str.strip().str.upper(),
            "category": original["category"],
            "rating": original["rating"],
            "source": "CG/OR review dataset",
        }
    )
    original_rows["label"] = original_rows["source_label"].map(LABEL_MAP)
    return original_rows


def load_amazon_dataset() -> pd.DataFrame:
    """Load the Amazon-labeled file separately for external evaluation."""
    if not AMAZON_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Amazon dataset not found at {AMAZON_DATA_PATH}. "
            "Put amazon_reviews.txt in the data folder."
        )
    amazon = pd.read_csv(AMAZON_DATA_PATH, sep="\t")
    required_amazon = {"LABEL", "RATING", "PRODUCT_CATEGORY", "REVIEW_TEXT"}
    missing = required_amazon - set(amazon.columns)
    if missing:
        raise ValueError(f"Amazon dataset is missing columns: {sorted(missing)}")

    amazon_rows = pd.DataFrame(
        {
            "text_": amazon["REVIEW_TEXT"],
            "source_label": amazon["LABEL"].astype("string").str.strip(),
            "category": amazon["PRODUCT_CATEGORY"],
            "rating": amazon["RATING"],
            "source": "Amazon labeled reviews",
        }
    )
    amazon_rows["label"] = amazon_rows["source_label"].map(LABEL_MAP)
    return amazon_rows


def load_raw_dataset() -> pd.DataFrame:
    """Load the two source datasets for the app's source-by-source statistics."""
    sources = [load_primary_dataset()]
    if AMAZON_DATA_PATH.exists():
        sources.append(load_amazon_dataset())
    return pd.concat(sources, ignore_index=True)


def analyze_dataset(data: pd.DataFrame) -> dict:
    """Summarize class balance, missing fields, and duplicate review text."""
    normalized_text = data["text_"].fillna("").map(clean_review)
    duplicate_groups = pd.DataFrame(
        {
            "review": normalized_text,
            "label": data["label"].fillna("Unknown").astype(str),
            "source": data["source"].fillna("Unknown").astype(str),
        }
    ).groupby("review", dropna=False)
    duplicate_label_counts = duplicate_groups["label"].nunique()
    duplicate_source_counts = duplicate_groups["source"].nunique()

    label_counts = data["label"].fillna("Unknown").value_counts(dropna=False)
    recognized_counts = [int(label_counts.get(label, 0)) for label in DISPLAY_LABELS]
    imbalance_ratio = (
        round(max(recognized_counts) / min(recognized_counts), 4)
        if min(recognized_counts) > 0
        else None
    )

    return {
        "row_count": int(len(data)),
        "missing_values_by_column": {
            str(column): int(count) for column, count in data.isna().sum().items()
        },
        "missing_text_rows": int(data["text_"].isna().sum()),
        "blank_text_rows": int(normalized_text.eq("").sum()),
        "unknown_label_rows": int(data["label"].isna().sum()),
        "exact_duplicate_rows": int(data.duplicated().sum()),
        "duplicate_raw_text_rows": int(data.duplicated(subset="text_").sum()),
        "duplicate_normalized_text_rows": int(normalized_text.duplicated().sum()),
        "conflicting_duplicate_text_groups": int((duplicate_label_counts > 1).sum()),
        "cross_source_duplicate_text_groups": int((duplicate_source_counts > 1).sum()),
        "raw_label_counts": {
            str(label): int(count) for label, count in label_counts.items()
        },
        "label_counts_by_source": {
            str(source): {
                str(label): int(count)
                for label, count in group["label"].value_counts(dropna=False).items()
            }
            for source, group in data.groupby("source", dropna=False)
        },
        "source_row_counts": {
            str(source): int(count)
            for source, count in data["source"].value_counts(dropna=False).items()
        },
        "fake_to_genuine_ratio": imbalance_ratio,
    }


def prepare_training_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean reviews and remove blank, ambiguous, or repeated normalized text."""
    clean_data = data[["text_", "label", "source"]].copy()
    clean_data["label"] = clean_data["label"].fillna("").astype(str).str.strip()
    clean_data = clean_data[clean_data["label"].isin(DISPLAY_LABELS)].copy()
    clean_data["review"] = clean_data["text_"].fillna("").map(clean_review)
    clean_data = clean_data[clean_data["review"].ne("")].copy()

    # Conflicting duplicates are ambiguous, so exclude every copy of that text.
    label_counts = clean_data.groupby("review")["label"].nunique()
    conflicting_texts = label_counts[label_counts > 1].index
    clean_data = clean_data[~clean_data["review"].isin(conflicting_texts)]

    # Drop overlaps across both source files before the train/test split.
    clean_data = clean_data.drop_duplicates(subset="review", keep="first")
    return clean_data[["review", "label", "source"]].rename(
        columns={"label": "target"}
    ).reset_index(drop=True)
