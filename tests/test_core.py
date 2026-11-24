"""Fast unit tests that need no trained model and no Streamlit."""

import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from review_logic import analyze_batch, assess_review_quality, highlight_review, influential_terms
from utils import clean_review, prepare_training_data, sanitize_csv_cell, sanitize_csv_frame

GENUINE = [
    "works great and arrived on time, would buy again",
    "the lid cracked after nine days but support replaced it quickly",
    "not bad at all, battery lasts long and the screen is bright",
    "terrible fit, the zipper broke and I never wore it again",
] * 5
FAKE = [
    "love it, great quality and very pretty, highly recommend it",
    "this product exceeded my expectations in every single way",
    "perfect gift, my family loves it and it looks wonderful",
    "excellent value, would recommend to anyone looking for this",
] * 5


@pytest.fixture(scope="module")
def tiny_model():
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(GENUINE + FAKE)
    clf = LogisticRegression(max_iter=500).fit(X, ["Genuine"] * len(GENUINE) + ["Fake"] * len(FAKE))
    return clf, vec


def test_clean_review_strips_markup_and_urls():
    out = clean_review("<b>Great</b> &amp; cheap http://x.co/a  ")
    assert out == "Great & cheap URL"


def test_prepare_training_data_drops_conflicts_and_duplicates():
    raw = pd.DataFrame({
        "text_": ["same", "same", "dup", "dup", "ok", None],
        "label": ["Fake", "Genuine", "Fake", "Fake", "Genuine", "Fake"],
        "source": "s",
    })
    out = prepare_training_data(raw)
    assert sorted(out["review"]) == ["dup", "ok"]


def test_csv_injection_is_neutralized():
    assert sanitize_csv_cell("=HYPERLINK(1)") == "'=HYPERLINK(1)"
    assert sanitize_csv_cell("@cmd") == "'@cmd"
    assert sanitize_csv_cell("fine") == "fine"
    frame = sanitize_csv_frame(pd.DataFrame({"t": ["=1+1", "ok"], "n": [1, 2]}))
    assert frame["t"].tolist() == ["'=1+1", "ok"] and frame["n"].tolist() == [1, 2]


def test_short_text_is_unsure(tiny_model):
    clf, vec = tiny_model
    reason, _ = assess_review_quality("great", vec, vec.transform(["great"]), 0.9, 0.6)
    assert reason and "at least" in reason


def test_non_english_is_unsure(tiny_model):
    clf, vec = tiny_model
    text = "es un buen producto llegó rápido y funciona perfecto"
    reason, coverage = assess_review_quality(text, vec, vec.transform([text]), 0.9, 0.6)
    assert reason and coverage < 0.6


def test_one_letter_words_count_toward_minimum(tiny_model):
    clf, vec = tiny_model
    text = "I am a big fan of it"
    reason, _ = assess_review_quality(text, vec, vec.transform([text]), 0.9, 0.6)
    assert not (reason and "at least" in reason)


def test_low_confidence_is_unsure(tiny_model):
    clf, vec = tiny_model
    text = "works great and arrived on time would buy again"
    reason, _ = assess_review_quality(text, vec, vec.transform([text]), 0.55, 0.6)
    assert reason and "below" in reason


def test_highlight_escapes_html():
    out = highlight_review("<script>alert(1)</script> great product", [{"term": "great", "contribution": 1.0}], "Genuine")
    assert "<script>" not in out and "&lt;script&gt;" in out and "<mark" in out


def test_influential_terms_only_support_prediction(tiny_model):
    clf, vec = tiny_model
    terms = influential_terms("works great would buy again", "Genuine", clf, vec)
    assert terms and all(t["contribution"] > 0 for t in terms)


def test_batch_handles_empty_and_blank_rows(tiny_model):
    clf, vec = tiny_model
    frame = pd.DataFrame({"review": ["works great and arrived on time would buy again", None, "   "]})
    out = analyze_batch(frame, "review", clf, vec, 0.6)
    assert out["Model prediction"].notna().tolist() == [True, False, False]
    assert out["Status"].iloc[1].startswith("Skipped")
