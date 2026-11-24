"""Train the Logistic Regression review classifier (C chosen by CV on training rows)."""

import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from utils import (
    AMAZON_DATA_PATH,
    DISPLAY_LABELS,
    LABEL_MAP,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    VECTORIZER_PATH,
    analyze_dataset,
    load_amazon_dataset,
    load_primary_dataset,
    prepare_training_data,
)


RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
# Regularization strengths tried; chosen by 5-fold CV on training rows only.
C_GRID = [1, 4, 10, 30]
TFIDF_SETTINGS = {
    "max_features": 50_000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
}


def split_each_source(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make a stratified holdout from each source before joining partitions."""
    train_parts = []
    test_parts = []
    for _, source_rows in data.groupby("source", sort=True):
        train_part, test_part = train_test_split(
            source_rows,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=source_rows["target"],
        )
        train_parts.append(train_part)
        test_parts.append(test_part)
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def evaluate_predictions(actual, predicted) -> dict:
    """Calculate standard metrics with Fake/Genuine order fixed for the matrix."""
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision_macro": float(precision_score(
            actual, predicted, labels=DISPLAY_LABELS, average="macro", zero_division=0
        )),
        "recall_macro": float(recall_score(
            actual, predicted, labels=DISPLAY_LABELS, average="macro", zero_division=0
        )),
        "f1_macro": float(f1_score(
            actual, predicted, labels=DISPLAY_LABELS, average="macro", zero_division=0
        )),
        "precision_fake": float(precision_score(actual, predicted, pos_label="Fake")),
        "recall_fake": float(recall_score(actual, predicted, pos_label="Fake")),
        "f1_fake": float(f1_score(actual, predicted, pos_label="Fake")),
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=DISPLAY_LABELS
        ).tolist(),
        "predicted_class_counts": {
            str(label): int(count)
            for label, count in pd.Series(predicted).value_counts().items()
        },
        "classification_report": classification_report(
            actual,
            predicted,
            labels=DISPLAY_LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_by_source(source_values, actual, predicted) -> dict:
    """Report metrics separately for each dataset source."""
    sources = pd.Series(source_values).reset_index(drop=True)
    actual = pd.Series(actual).reset_index(drop=True)
    predicted = pd.Series(predicted).reset_index(drop=True)
    return {
        str(source): evaluate_predictions(actual[sources.eq(source)], predicted[sources.eq(source)])
        for source in sorted(sources.unique())
    }


def amazon_reference_baselines(amazon_raw: pd.DataFrame, amazon_external: pd.DataFrame) -> dict:
    """Context for the Amazon score: how learnable is that source at all?

    1) a text-only model trained and tested inside Amazon (80/20 split), and
    2) the VERIFIED_PURCHASE column alone. The latter is a dataset shortcut, not
    evidence of fakery; it is reported only to show text is a weak signal there.
    These numbers never touch the deployed model.
    """
    train_part, test_part = train_test_split(
        amazon_external, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=amazon_external["target"],
    )
    vec = TfidfVectorizer(**TFIDF_SETTINGS)
    clf = LogisticRegression(max_iter=3_000, random_state=RANDOM_STATE)
    clf.fit(vec.fit_transform(train_part["review"]), train_part["target"])
    in_domain = float(clf.score(vec.transform(test_part["review"]), test_part["target"]))

    verified = pd.read_csv(AMAZON_DATA_PATH, sep="\t")
    verified_only = float(
        (verified["VERIFIED_PURCHASE"].map({"Y": "Genuine", "N": "Fake"})
         == verified["LABEL"].map(LABEL_MAP)).mean()
    )
    return {
        "amazon_text_only_in_domain_accuracy": in_domain,
        "verified_purchase_only_accuracy": verified_only,
        "note": "Reference only; not used for training or selection of the deployed model.",
    }


def main() -> None:
    # This project deploys Logistic Regression, matching the original brief.
    # The optional compare_models.py file compares other classifiers separately.
    raw_data = load_primary_dataset()
    dataset_summary = analyze_dataset(raw_data)
    data = prepare_training_data(raw_data)
    train_rows, test_rows = split_each_source(data)
    X_train, y_train = train_rows["review"], train_rows["target"]
    X_test, y_test = test_rows["review"], test_rows["target"]

    print("Dataset check:")
    print(json.dumps(dataset_summary, indent=2))
    print("Cleaned training-source rows:", len(data))
    print("Training labels:", y_train.value_counts().reindex(DISPLAY_LABELS).to_dict())
    print("Test labels:", y_test.value_counts().reindex(DISPLAY_LABELS).to_dict())

    # TF-IDF lives inside the pipeline so every CV fold learns its vocabulary from
    # that fold's training rows only. The test split is never used to pick C.
    model_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(**TFIDF_SETTINGS)),
            ("classifier", LogisticRegression(max_iter=3_000, random_state=RANDOM_STATE)),
        ]
    )
    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    search = GridSearchCV(
        model_pipeline,
        {"classifier__C": C_GRID},
        cv=cross_validation,
        scoring={
            "accuracy": "accuracy",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "f1_macro": "f1_macro",
        },
        refit=False,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    results = search.cv_results_
    best_index = int(np.argmin(results["rank_test_f1_macro"]))
    best_c = float(results["param_classifier__C"][best_index])
    cv_summary = {
        f"{metric}_mean": float(results[f"mean_test_{metric}"][best_index])
        for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    }
    cv_summary["f1_macro_std"] = float(results["std_test_f1_macro"][best_index])
    cv_by_c = {
        str(float(c)): float(score)
        for c, score in zip(results["param_classifier__C"], results["mean_test_f1_macro"])
    }
    print(f"CV macro F1 by C: {cv_by_c}; selected C={best_c}")

    # Fit the final vocabulary and classifier using only training rows.
    vectorizer = TfidfVectorizer(**TFIDF_SETTINGS)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    classifier = LogisticRegression(C=best_c, max_iter=3_000, random_state=RANDOM_STATE)
    classifier.fit(X_train_tfidf, y_train)
    X_test_tfidf = vectorizer.transform(X_test)
    test_predictions = classifier.predict(X_test_tfidf)
    test_metrics = {
        **evaluate_predictions(y_test, test_predictions),
        "by_source": evaluate_by_source(
            test_rows["source"], y_test, test_predictions
        ),
    }

    # The Amazon file remains external: transform it with this vocabulary only.
    amazon_raw = load_amazon_dataset()
    amazon_data = prepare_training_data(amazon_raw)
    primary_reviews = set(data["review"])
    amazon_overlap = amazon_data["review"].isin(primary_reviews)
    amazon_external = amazon_data.loc[~amazon_overlap].reset_index(drop=True)
    amazon_predictions = classifier.predict(
        vectorizer.transform(amazon_external["review"])
    )
    amazon_metrics = evaluate_predictions(
        amazon_external["target"], amazon_predictions
    )

    reference_baselines = amazon_reference_baselines(amazon_raw, amazon_external)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    report = {
        "dataset": dataset_summary,
        "cleaned_rows": int(len(data)),
        "cleaned_class_counts": {
            str(label): int(count)
            for label, count in data["target"].value_counts().items()
        },
        "cleaned_fake_to_genuine_ratio": round(
            data["target"].value_counts().get("Fake", 0)
            / data["target"].value_counts().get("Genuine", 1),
            4,
        ),
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified_by_label_within_each_source": True,
            "train_rows": int(len(train_rows)),
            "test_rows": int(len(test_rows)),
            "train_class_counts": {
                str(label): int(count) for label, count in y_train.value_counts().items()
            },
            "test_class_counts": {
                str(label): int(count) for label, count in y_test.value_counts().items()
            },
        },
        "cross_validation": {
            "folds": CV_FOLDS,
            "selection_metric": "f1_macro, 5-fold CV on training rows; only C is tuned",
            "cv_f1_macro_by_c": cv_by_c,
            "scores": cv_summary,
            "models": {
                "Logistic Regression": {
                    "cv_f1_macro": cv_summary["f1_macro_mean"],
                    "test": test_metrics,
                    "parameters": {
                        "model": "Logistic Regression",
                        "C": best_c,
                        "tfidf": {
                            **TFIDF_SETTINGS,
                            "ngram_range": list(TFIDF_SETTINGS["ngram_range"]),
                        },
                    },
                }
            },
        },
        "external_evaluation": {
            "source": "Amazon labeled reviews",
            "used_for_training_or_selection": False,
            "dataset": analyze_dataset(amazon_raw),
            "cleaned_rows_before_cross_source_deduplication": int(len(amazon_data)),
            "overlapping_rows_removed": int(amazon_overlap.sum()),
            "evaluated_rows": int(len(amazon_external)),
            "test": amazon_metrics,
            "reference_baselines": reference_baselines,
        },
        "label_mapping": {
            "CG": "Fake",
            "OR": "Genuine",
            "__label1__": "Fake",
            "__label2__": "Genuine",
        },
        "model_name": "Logistic Regression",
        "model_parameters": {
            "model": "Logistic Regression",
            "C": best_c,
            "class_weight": None,
            "tfidf": {
                **TFIDF_SETTINGS,
                "ngram_range": list(TFIDF_SETTINGS["ngram_range"]),
            },
        },
        "test": test_metrics,
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "training": {
            "tfidf": {
                **TFIDF_SETTINGS,
                "ngram_range": list(TFIDF_SETTINGS["ngram_range"]),
            },
            "classifier": f"LogisticRegression(C={best_c}, max_iter=3000)",
            "cv_only_uses_training_rows": True,
            "test_split_used_for_selection": False,
            "numpy_version": np.__version__,
        },
        "saved_model": MODEL_PATH.name,
        "saved_vectorizer": VECTORIZER_PATH.name,
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Amazon reference baselines:", json.dumps(reference_baselines))
    print("\nFive-fold CV summary:", json.dumps(cv_summary, indent=2))
    print("\nHeld-out classification report:")
    print(classification_report(
        y_test,
        test_predictions,
        labels=DISPLAY_LABELS,
        digits=4,
        zero_division=0,
    ))
    print("Confusion matrix (actual rows/predicted columns; Fake, Genuine):")
    print(test_metrics["confusion_matrix"])
    print("Predicted class counts:", test_metrics["predicted_class_counts"])
    print("\nAmazon external classification report (not used for fitting):")
    print(classification_report(
        amazon_external["target"],
        amazon_predictions,
        labels=DISPLAY_LABELS,
        digits=4,
        zero_division=0,
    ))
    print("Amazon confusion matrix:", amazon_metrics["confusion_matrix"])
    print("Saved model:", MODEL_PATH)
    print("Saved vectorizer:", VECTORIZER_PATH)
    print("Saved metrics:", METRICS_PATH)

if __name__ == "__main__":
    main()
