"""Prediction helpers for the app. No Streamlit import, so they are unit-testable."""

import html
import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from utils import MIN_VOCAB_COVERAGE, MIN_WORDS, clean_review


WORD_PATTERN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")


def probability_details(model, features) -> tuple[dict[str, float], float | None]:
    """Return class probabilities and the probability of the predicted class."""
    if not hasattr(model, "predict_proba"):
        return {}, None

    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)
    by_class = {
        str(label): float(probabilities[index])
        for index, label in enumerate(classes)
    }
    predicted = str(model.predict(features)[0])
    return by_class, by_class.get(predicted)


def assess_review_quality(review: str, vectorizer, features,
                          confidence: float | None,
                          confidence_threshold: float) -> tuple[str | None, float]:
    """Abstain on very short, unfamiliar, or low-confidence text."""
    # Count what the user typed (including one-letter words), not model tokens.
    if len(WORD_PATTERN.findall(review)) < MIN_WORDS:
        return f"Enter at least {MIN_WORDS} words for a useful estimate.", 0.0

    analyzer = vectorizer.build_analyzer()
    words = [term for term in analyzer(review) if " " not in term]
    if not words:
        return "No recognizable words were found in this text.", 0.0

    vocabulary = vectorizer.vocabulary_
    known_words = sum(word in vocabulary for word in words)
    coverage = known_words / len(words) if words else 0.0
    if features.nnz == 0 or coverage < MIN_VOCAB_COVERAGE:
        return (
            "Many words are outside this model's training vocabulary (it only "
            "knows English product-review text). Try an English review.",
            coverage,
        )
    if confidence is not None and confidence < confidence_threshold:
        return (
            f"The model confidence is below your {confidence_threshold:.0%} threshold.",
            coverage,
        )
    return None, coverage


def influential_terms(review: str, predicted_label: str, model, vectorizer,
                      limit: int = 6) -> list[dict[str, float | str]]:
    """Rank words/phrases using the deployed Logistic Regression weights."""
    review_features = vectorizer.transform([review])
    classes = list(model.classes_)
    predicted_index = classes.index(predicted_label)
    # For binary Logistic Regression, positive coefficients favor classes_[1].
    direction = 1 if predicted_index == 1 else -1
    feature_weights = model.coef_[0] * direction

    row = review_features.tocsr()
    contributions = row.data * feature_weights[row.indices]
    feature_names = vectorizer.get_feature_names_out()
    negations = {"not", "no", "nor", "never"}
    ranked = [
        {"term": str(feature_names[index]), "contribution": float(score)}
        for index, score in zip(row.indices, contributions)
        if score > 0
        and any(
            word not in ENGLISH_STOP_WORDS or word in negations
            for word in str(feature_names[index]).split()
        )
    ]
    ranked.sort(key=lambda item: item["contribution"], reverse=True)
    return ranked[:limit]


def highlight_review(review: str, terms: list[dict], predicted_label: str) -> str:
    """Highlight matching words/phrases, safely escaping the user's review text."""
    css_class = "fake-support" if predicted_label == "Fake" else "genuine-support"
    matches = []

    for term_item in terms:
        term = str(term_item["term"])
        flexible_term = r"\s+".join(re.escape(part) for part in term.split())
        pattern = re.compile(rf"(?<!\w){flexible_term}(?!\w)", flags=re.IGNORECASE)
        for match in pattern.finditer(review):
            matches.append((match.start(), match.end(), float(term_item["contribution"])))

    # Choose high-contribution phrases first, then keep non-overlapping highlights.
    selected = []
    for start, end, contribution in sorted(matches, key=lambda item: item[2], reverse=True):
        if all(end <= old_start or start >= old_end for old_start, old_end in selected):
            selected.append((start, end))
    selected.sort()

    parts = []
    cursor = 0
    for start, end in selected:
        before = html.escape(review[cursor:start]).replace("\n", "<br>")
        highlighted = html.escape(review[start:end])
        parts.append(before)
        parts.append(f'<mark class="{css_class}">{highlighted}</mark>')
        cursor = end
    parts.append(html.escape(review[cursor:]).replace("\n", "<br>"))
    return "".join(parts)


def analyze_batch(frame: pd.DataFrame, text_column: str, model, vectorizer,
                  confidence_threshold: float) -> pd.DataFrame:
    """Predict every non-empty review in a CSV and retain empty rows as skipped."""
    result = frame.copy()
    original_text = result[text_column]
    cleaned_reviews = [
        "" if pd.isna(value) else clean_review(value)
        for value in original_text
    ]
    valid_positions = [
        index for index, value in enumerate(cleaned_reviews) if value
    ]

    predictions: list[str | None] = [None] * len(result)
    app_results: list[str | None] = [None] * len(result)
    confidences: list[float | None] = [None] * len(result)
    status = ["Skipped: empty review"] * len(result)

    if valid_positions:
        features = vectorizer.transform([cleaned_reviews[index] for index in valid_positions])
        batch_predictions = model.predict(features)
        probability_matrix = (
            model.predict_proba(features) if hasattr(model, "predict_proba") else None
        )
        classes = list(model.classes_)

        for batch_index, row_position in enumerate(valid_positions):
            label = str(batch_predictions[batch_index])
            predictions[row_position] = label
            if probability_matrix is not None:
                confidence = float(
                    probability_matrix[batch_index][classes.index(label)]
                )
                confidences[row_position] = confidence
            else:
                confidence = None
            reason, coverage = assess_review_quality(
                cleaned_reviews[row_position],
                vectorizer,
                features[batch_index],
                confidence,
                confidence_threshold,
            )
            app_results[row_position] = "Unsure" if reason else label
            status[row_position] = reason or (
                f"Usable text; {coverage:.0%} of its words are in the model vocabulary."
            )

    result["Model prediction"] = predictions
    result["App result"] = app_results
    result["Prediction confidence"] = confidences
    result["Status"] = status
    return result
