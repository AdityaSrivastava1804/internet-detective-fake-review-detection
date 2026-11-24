"""Optional CV-only comparison; it never replaces the deployed model."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from train_model import C_GRID, RANDOM_STATE, TFIDF_SETTINGS, split_each_source
from utils import load_primary_dataset, prepare_training_data



def main() -> None:
    """Compare three classifiers using CV on training data only."""
    data = prepare_training_data(load_primary_dataset())
    train_rows, _ = split_each_source(data)
    X_train, y_train = train_rows["review"], train_rows["target"]

    candidates = {
        **{f"Logistic Regression (C={c})": LogisticRegression(C=c, max_iter=3_000, random_state=RANDOM_STATE)
           for c in C_GRID},
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Linear SVM": LinearSVC(random_state=42),
    }
    fold_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }
    comparison = []

    for name, classifier in candidates.items():
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(**TFIDF_SETTINGS)),
                ("classifier", classifier),
            ]
        )
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=fold_strategy,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        comparison.append(
            {
                "Model": name,
                "CV accuracy": scores["test_accuracy"].mean(),
                "CV precision (macro)": scores["test_precision_macro"].mean(),
                "CV recall (macro)": scores["test_recall_macro"].mean(),
                "CV F1 (macro)": scores["test_f1_macro"].mean(),
            }
        )

    result = pd.DataFrame(comparison).sort_values(
        "CV F1 (macro)", ascending=False
    )
    print("Five-fold CV comparison on the training split only:")
    print(result.to_string(index=False, formatters={
        column: (lambda value: f"{value:.3f}")
        for column in result.columns if column != "Model"
    }))
    print("\nThis optional script does not alter the deployed model or test metrics.")


if __name__ == "__main__":
    main()
