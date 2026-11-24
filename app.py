"""Streamlit interface for the Internet Detective review classifier."""

import io
import json
import platform

import joblib
import pandas as pd
import sklearn
import streamlit as st

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    # Streamlit's built-in charts keep the app usable if Matplotlib is missing.
    plt = None

from utils import (
    AMAZON_DATA_PATH,
    DATA_PATH,
    DISPLAY_LABELS,
    METRICS_PATH,
    MODEL_PATH,
    VECTORIZER_PATH,
    clean_review,
    load_raw_dataset,
    sanitize_csv_frame,
)
from review_logic import (
    analyze_batch,
    assess_review_quality,
    highlight_review,
    influential_terms,
    probability_details,
)


LOW_CONFIDENCE_DEFAULT = 0.60

st.set_page_config(
    page_title="Internet Detective | Review Classifier",
    page_icon="🕵️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(145deg, #f6f8fc 0%, #eef4ff 100%); }
      .hero {
        padding: 2.1rem 2.3rem; border-radius: 22px; color: #fff;
        background: linear-gradient(115deg, #102a43 0%, #176b87 58%, #3aafa9 100%);
        box-shadow: 0 16px 38px rgba(16, 42, 67, .16); margin-bottom: 1.4rem;
      }
      .hero h1 { color: #fff; margin: 0 0 .35rem 0; font-size: 2.5rem; }
      .hero p { color: #e4f7f5; font-size: 1.08rem; margin: 0; }
      .section-title { color: #102a43; font-weight: 700; margin: .5rem 0 .2rem; }
      div[data-testid="stMetric"] {
        background: rgba(255,255,255,.85); border: 1px solid #dbe7f2;
        padding: 14px 18px; border-radius: 15px;
      }
      .note { color: #486581; font-size: .92rem; }
      .review-highlight {
        background: #ffffff; border: 1px solid #dbe7f2; border-radius: 12px;
        padding: 1rem; line-height: 1.8; white-space: normal;
      }
      mark.fake-support { background: #ffd6cc; color: #762d1c; border-radius: 4px; }
      mark.genuine-support { background: #c9f3e7; color: #145842; border-radius: 4px; }
    </style>
    <div class="hero">
      <h1>🕵️ Internet Detective</h1>
      <p>An educational classifier for labeled review datasets. It is not a reliable detector for every review source.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_files(model_modified: int, vectorizer_modified: int):
    """Load artifacts; timestamps refresh them after local training."""
    del model_modified, vectorizer_modified
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


@st.cache_data
def load_metrics(metrics_modified: int):
    """Read model metrics again whenever training rewrites the report."""
    del metrics_modified
    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


@st.cache_data
def load_dataset_snapshot(primary_modified: int, amazon_modified: int):
    """Avoid rereading the CSV and TSV on every Streamlit rerun."""
    del primary_modified, amazon_modified
    return load_raw_dataset()


def file_modified_time(path) -> int:
    """Return a cache key for a file, or zero when optional files are absent."""
    return path.stat().st_mtime_ns if path.exists() else 0


def show_grouped_chart(chart_data: pd.DataFrame, title: str, y_label: str):
    """Draw a simple Matplotlib chart, with a Streamlit fallback."""
    if chart_data.empty:
        st.info("There is no data to chart.")
        return
    if plt is None:
        st.bar_chart(chart_data)
        return

    figure, axis = plt.subplots(figsize=(8, 4))
    chart_data.plot(kind="bar", ax=axis, color=["#e76f51", "#2a9d8f", "#457b9d"])
    axis.set_title(title)
    axis.set_ylabel(y_label)
    axis.set_xlabel("")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(title="")
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def show_confusion_matrix(result: dict, title: str):
    """Show actual-versus-predicted counts for one evaluated dataset."""
    matrix = result["confusion_matrix"]
    if plt is None:
        st.dataframe(
            pd.DataFrame(matrix, index=DISPLAY_LABELS, columns=DISPLAY_LABELS),
            width="stretch",
        )
        return

    figure, axis = plt.subplots(figsize=(5, 3.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(DISPLAY_LABELS)), DISPLAY_LABELS)
    axis.set_yticks(range(len(DISPLAY_LABELS)), DISPLAY_LABELS)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    axis.set_title(title)
    for row in range(len(DISPLAY_LABELS)):
        for column in range(len(DISPLAY_LABELS)):
            axis.text(column, row, f"{matrix[row][column]:,}", ha="center", va="center")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def class_metrics_table(result: dict) -> pd.DataFrame:
    """Turn the report's per-class values into a compact readable table."""
    report = result["classification_report"]
    return pd.DataFrame(
        [
            {
                "Class": label,
                "Precision": report[label]["precision"],
                "Recall": report[label]["recall"],
                "F1-score": report[label]["f1-score"],
                "Support": int(report[label]["support"]),
            }
            for label in DISPLAY_LABELS
        ]
    ).style.format(
        {
            "Precision": "{:.1%}",
            "Recall": "{:.1%}",
            "F1-score": "{:.1%}",
        }
    )


try:
    data = load_dataset_snapshot(
        file_modified_time(DATA_PATH),
        file_modified_time(AMAZON_DATA_PATH),
    )
    metrics = load_metrics(file_modified_time(METRICS_PATH))
    model, vectorizer = load_model_files(
        MODEL_PATH.stat().st_mtime_ns,
        VECTORIZER_PATH.stat().st_mtime_ns,
    )
except FileNotFoundError as error:
    st.error(f"{error}\n\nRun `python3 train_model.py` from the project folder first.")
    st.stop()
except Exception as error:
    st.error(
        f"Could not load the project files: {error}. Run `python3 train_model.py` "
        "in this environment to rebuild compatible model files."
    )
    st.stop()

saved_sklearn_version = metrics.get("versions", {}).get("scikit_learn")
saved_python_version = metrics.get("versions", {}).get("python")
if (
    (saved_sklearn_version and saved_sklearn_version != sklearn.__version__)
    or (saved_python_version and saved_python_version.split(".")[:2] != platform.python_version().split(".")[:2])
):
    st.warning(
        "The model files were created in a different Python or scikit-learn "
        "environment. Retrain here (about a minute) so they match your installed "
        "libraries."
    )
    if st.button("Retrain model now", key="retrain_now"):
        import train_model

        with st.spinner("Training... this takes about a minute."):
            train_model.main()
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

external = metrics.get("external_evaluation")
if external:
    amazon_result = external["test"]
    st.error(
        "Generalization warning — on the separate Amazon dataset, this model has "
        f"{amazon_result['accuracy']:.1%} accuracy and "
        f"{amazon_result['recall_fake']:.1%} Fake recall. It misses most Amazon "
        "Fake-labeled reviews. Treat its predictions as experimental outside the "
        "CG/OR training source."
    )


if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None
if "feedback_entries" not in st.session_state:
    st.session_state.feedback_entries = []
if "feedback_message" not in st.session_state:
    st.session_state.feedback_message = ""
if "batch_result" not in st.session_state:
    st.session_state.batch_result = None

confidence_threshold = st.sidebar.slider(
    "Show Unsure below this confidence",
    min_value=0.55,
    max_value=0.95,
    value=LOW_CONFIDENCE_DEFAULT,
    step=0.05,
    help="Lower-confidence results are shown as Unsure. The raw Logistic Regression label remains visible in the result details.",
)
st.sidebar.caption(
    "Short text and non-English text are shown as Unsure even when the raw model score looks high."
)
st.sidebar.warning(
    "Confidence is NOT a domain-shift detector: on Amazon reviews the model is "
    "confidently wrong. Never treat a high score as proof."
)


left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.markdown('<div class="section-title">Try one review</div>', unsafe_allow_html=True)
    st.write("Paste a product review below. The classifier uses its text only.")
    review_text = st.text_area(
        "Review text",
        placeholder="For example: The item arrived quickly and works just as described...",
        height=170,
        label_visibility="collapsed",
        key="single_review_text",
    )
    analyze_clicked = st.button(
        "Analyze review", type="primary", width="stretch", key="analyze_one"
    )

    if analyze_clicked:
        cleaned_review = clean_review(review_text)
        if not cleaned_review:
            st.session_state.latest_prediction = None
            st.warning("Enter some review text first.")
        else:
            features = vectorizer.transform([cleaned_review])
            predicted_label = str(model.predict(features)[0])
            probabilities, confidence = probability_details(model, features)
            unsure_reason, vocabulary_coverage = assess_review_quality(
                cleaned_review,
                vectorizer,
                features,
                confidence,
                confidence_threshold,
            )
            terms = influential_terms(
                cleaned_review, predicted_label, model, vectorizer
            )
            st.session_state.latest_prediction = {
                "review": review_text,
                "cleaned_review": cleaned_review,
                "label": predicted_label,
                "confidence": confidence,
                "probabilities": probabilities,
                "terms": terms,
                "app_result": "Unsure" if unsure_reason else predicted_label,
                "unsure_reason": unsure_reason,
                "vocabulary_coverage": vocabulary_coverage,
            }
            st.session_state.feedback_message = ""

    latest = st.session_state.latest_prediction
    if latest:
        current_features = vectorizer.transform([latest["cleaned_review"]])
        latest["unsure_reason"], latest["vocabulary_coverage"] = assess_review_quality(
            latest["cleaned_review"],
            vectorizer,
            current_features,
            latest["confidence"],
            confidence_threshold,
        )
        latest["app_result"] = (
            "Unsure" if latest["unsure_reason"] else latest["label"]
        )
        st.markdown("#### Latest result")
        st.warning(
            "This result reflects patterns from the CG/OR dataset, where Fake means "
            "computer-generated and Genuine means original. The model has weak "
            "performance on Amazon reviews and may be wrong on other sources."
        )
        result_col, confidence_col = st.columns(2)
        result_col.metric("App result", latest["app_result"])

        if latest["unsure_reason"]:
            st.warning(
                "Unsure — " + latest["unsure_reason"] +
                f" Raw classifier preference: {latest['label']}"
                + (f" ({latest['confidence']:.1%})" if latest["confidence"] is not None else ".")
            )
            if latest["confidence"] is not None:
                confidence_col.metric("Raw model score · not reliable here", f"{latest['confidence']:.1%}")
        elif latest["confidence"] is not None:
            confidence_col.metric(
                "Prediction confidence", f"{latest['confidence']:.1%}"
            )
            st.progress(latest["confidence"])
            st.caption(
                f"Fake likelihood: {latest['probabilities'].get('Fake', 0):.1%} · "
                f"Genuine likelihood: {latest['probabilities'].get('Genuine', 0):.1%}"
            )
        else:
            confidence_col.info("This model does not provide probability estimates.")

        st.markdown("**Words and phrases that support this prediction**")
        if latest["app_result"] == "Unsure":
            st.info("The text did not meet the minimum quality checks for an explanation.")
        elif latest["terms"]:
            highlight_html = highlight_review(
                latest["review"], latest["terms"], latest["label"]
            )
            st.markdown(
                f'<div class="review-highlight">{highlight_html}</div>',
                unsafe_allow_html=True,
            )
            term_summary = ", ".join(
                f"`{item['term']}`" for item in latest["terms"]
            )
            st.caption(f"Top contributing features: {term_summary}")
        else:
            st.info(
                "No review words from the model vocabulary strongly supported this "
                "prediction. The decision may depend on other features or model bias."
            )
        st.caption(
            "Highlights show associations learned from training examples; they do not "
            "prove that a review is deceptive."
        )

        st.markdown("**Record a correction for export**")
        st.caption(
            "Use the dataset's label meaning: Fake means computer-generated; Genuine "
            "means original. Corrections are only collected for CSV export; they do "
            "not change or retrain the model."
        )
        feedback_left, feedback_right = st.columns(2)
        correction_choice = None
        if feedback_left.button("Correct label: Fake", key="correct_fake"):
            correction_choice = "Fake"
        if feedback_right.button("Correct label: Genuine", key="correct_genuine"):
            correction_choice = "Genuine"

        if correction_choice:
            entry = {
                "Review": latest["review"],
                "Model prediction": latest["label"],
                "Corrected label": correction_choice,
            }
            if entry in st.session_state.feedback_entries:
                st.session_state.feedback_message = "That correction is already in the export list."
            else:
                st.session_state.feedback_entries.append(entry)
                st.session_state.feedback_message = (
                    "Correction added to this session's export list. The model and its "
                    "prediction were not changed."
                )

        if st.session_state.feedback_message:
            st.info(st.session_state.feedback_message)

with right:
    st.markdown('<div class="section-title">About this model</div>', unsafe_allow_html=True)
    st.write(
        f"The selected model is **{metrics['model_name']}**, trained on the original "
        "CG/OR training split. TF-IDF turns "
        "review words and phrases into numbers; "
        "the classifier uses those features to choose between Fake and Genuine."
    )
    st.markdown(
        '<div class="note">For the CG/OR source, CG maps to Fake (computer-generated) '
        'and OR maps to Genuine (original). The Amazon file is held out for external '
        'evaluation only: __label1__ maps to Fake and __label2__ to Genuine. Category '
        'and rating are not model inputs.</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "The confidence threshold may show Unsure. It does not change the underlying "
        "Logistic Regression label."
    )


with st.expander("Analyze many reviews from a CSV", expanded=False):
    st.write(
        "Upload a CSV with a column containing review text. The app adds the raw model "
        "label, an app result (Fake, Genuine, or Unsure), confidence, and a reason. "
        "Empty rows are kept and marked skipped."
    )
    uploaded_file = st.file_uploader(
        "Choose a CSV file", type=["csv"], key="batch_csv_upload"
    )
    if uploaded_file is not None:
        try:
            uploaded_frame = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
            if uploaded_frame.empty:
                st.warning("This CSV has headers but no review rows.")
            else:
                text_columns = [
                    column for column in uploaded_frame.columns
                    if uploaded_frame[column].dtype == "object"
                    or pd.api.types.is_string_dtype(uploaded_frame[column])
                ]
                if not text_columns:
                    st.error("I couldn't find a text column in this CSV.")
                else:
                    suggested_column = next(
                        (
                            column for column in text_columns
                            if any(word in str(column).lower() for word in ["review", "text"])
                        ),
                        text_columns[0],
                    )
                    default_index = text_columns.index(suggested_column)
                    selected_column = st.selectbox(
                        "Which column contains the reviews?",
                        options=text_columns,
                        index=default_index,
                        key="batch_text_column",
                    )
                    if st.button("Predict CSV reviews", type="primary", key="predict_csv"):
                        with st.spinner("Analyzing reviews..."):
                            batch_predictions = analyze_batch(
                                uploaded_frame,
                                selected_column,
                                model,
                                vectorizer,
                                confidence_threshold,
                            )
                        st.session_state.batch_result = {
                            "filename": uploaded_file.name,
                            "text_column": selected_column,
                            "frame": batch_predictions,
                        }

                    batch_result = st.session_state.batch_result
                    if (
                        batch_result
                        and batch_result["filename"] == uploaded_file.name
                        and batch_result["text_column"] == selected_column
                    ):
                        output_frame = batch_result["frame"]
                        predicted_count = int(output_frame["Model prediction"].notna().sum())
                        unsure_count = int(output_frame["App result"].eq("Unsure").sum())
                        st.success(
                            f"Analyzed {predicted_count:,} reviews; "
                            f"{unsure_count:,} marked Unsure."
                        )
                        st.warning(
                            "These results reflect CG/OR dataset patterns. The model "
                            "has weak performance on Amazon reviews and may be wrong "
                            "for other sources."
                        )
                        st.dataframe(output_frame, width="stretch", hide_index=True)
                        csv_bytes = sanitize_csv_frame(output_frame).to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download predictions CSV",
                            data=csv_bytes,
                            file_name=f"predictions_{uploaded_file.name}",
                            mime="text/csv",
                            key="download_predictions",
                        )
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
            st.error(f"Could not read this CSV: {error}")


st.divider()
st.markdown('<div class="section-title">Dataset and model performance</div>', unsafe_allow_html=True)
source_counts = pd.crosstab(data["source"], data["label"]).reindex(
    columns=DISPLAY_LABELS, fill_value=0
)
source_counts.index.name = "Dataset source"

training_source = "CG/OR review dataset"
amazon_source = "Amazon labeled reviews"
training_counts = source_counts.loc[training_source] if training_source in source_counts.index else pd.Series(dtype=int)
amazon_counts = source_counts.loc[amazon_source] if amazon_source in source_counts.index else pd.Series(dtype=int)
stats = st.columns(5)
stats[0].metric("CG/OR training reviews", f"{int(training_counts.sum()):,}")
stats[1].metric("CG / Fake", f"{int(training_counts.get('Fake', 0)):,}")
stats[2].metric("OR / Genuine", f"{int(training_counts.get('Genuine', 0)):,}")
stats[3].metric("Amazon external reviews", f"{int(amazon_counts.sum()):,}")
stats[4].metric("Corrections queued", f"{len(st.session_state.feedback_entries):,}")
st.caption(
    "Counts are shown by source because the datasets use different label definitions. "
    "CG means computer-generated in the training data; Amazon labels come from a separate source."
)

source_performance = []
for source, result in metrics["test"].get("by_source", {}).items():
    source_performance.append(
        {
            "Dataset": source + " · held-out test",
            "Accuracy": result["accuracy"],
            "Macro F1": result["f1_macro"],
            "Fake recall": result["recall_fake"],
            "Rows": int(sum(
                result["classification_report"][label]["support"]
                for label in DISPLAY_LABELS
            )),
        }
    )
external = metrics.get("external_evaluation")
if external:
    external_result = external["test"]
    source_performance.append(
        {
            "Dataset": "Amazon labeled reviews · external",
            "Accuracy": external_result["accuracy"],
            "Macro F1": external_result["f1_macro"],
            "Fake recall": external_result["recall_fake"],
            "Rows": external["evaluated_rows"],
        }
    )
performance_frame = pd.DataFrame(source_performance)

balance_tab, rating_tab, performance_tab = st.tabs(
    ["Label balance", "Ratings by source", "Performance by source"]
)
with balance_tab:
    st.dataframe(source_counts, width="stretch")
    show_grouped_chart(source_counts, "Fake and Genuine labels by dataset", "Reviews")
with rating_tab:
    numeric_ratings = data.copy()
    numeric_ratings["rating_number"] = pd.to_numeric(
        numeric_ratings["rating"], errors="coerce"
    )
    rating_counts = (
        numeric_ratings.dropna(subset=["rating_number"])
        .groupby(["rating_number", "source"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    rating_counts.index.name = "Star rating"
    if rating_counts.empty:
        st.info("Neither source has numeric ratings to compare.")
    else:
        st.dataframe(rating_counts, width="stretch")
        show_grouped_chart(rating_counts, "Rating distribution by source", "Reviews")
with performance_tab:
    if performance_frame.empty:
        st.info("There are no per-source evaluation metrics in the saved report.")
    else:
        st.dataframe(
            performance_frame.style.format(
                {"Accuracy": "{:.1%}", "Macro F1": "{:.1%}", "Fake recall": "{:.1%}"}
            ),
            width="stretch",
            hide_index=True,
        )
        chart_frame = performance_frame.set_index("Dataset")[[
            "Accuracy", "Macro F1", "Fake recall"
        ]]
        show_grouped_chart(
            chart_frame, "Model results by dataset source", "Score"
        )
        if external and external.get("reference_baselines"):
            ref = external["reference_baselines"]
            st.info(
                "Context for the Amazon number: a text-only model trained on Amazon "
                f"itself reaches only {ref['amazon_text_only_in_domain_accuracy']:.1%}, while "
                "the VERIFIED_PURCHASE column alone reaches "
                f"{ref['verified_purchase_only_accuracy']:.1%}. The latter is a dataset "
                "shortcut, not evidence of fakery. Text alone is a weak signal on Amazon."
            )
        if external:
            st.warning(
                "Amazon is an external check. It was not used for training or model "
                "selection, and its labels and review distribution differ from the "
                "CG/OR training data."
            )

        evaluated_sources = [
            (source + " · held-out test", result)
            for source, result in metrics["test"].get("by_source", {}).items()
        ]
        if external:
            evaluated_sources.append(
                ("Amazon labeled reviews · external", external["test"])
            )
        for source_name, source_result in evaluated_sources:
            with st.expander(f"Detailed metrics · {source_name}", expanded=True):
                detail_columns = st.columns(4)
                detail_columns[0].metric("Accuracy", f"{source_result['accuracy']:.1%}")
                detail_columns[1].metric(
                    "Precision · macro", f"{source_result['precision_macro']:.1%}"
                )
                detail_columns[2].metric(
                    "Recall · macro", f"{source_result['recall_macro']:.1%}"
                )
                detail_columns[3].metric(
                    "F1-score · macro", f"{source_result['f1_macro']:.1%}"
                )
                matrix_col, class_col = st.columns([0.9, 1.1])
                with matrix_col:
                    show_confusion_matrix(source_result, "Confusion matrix")
                with class_col:
                    st.markdown("**Per-class precision, recall, and F1**")
                    st.dataframe(
                        class_metrics_table(source_result),
                        width="stretch",
                        hide_index=True,
                    )

st.markdown("**Logistic Regression cross-validation results**")
comparison_rows = []
for family, result in metrics["cross_validation"]["models"].items():
    test = result["test"]
    comparison_rows.append(
        {
            "Model": family,
            "CV macro F1": result["cv_f1_macro"],
            "Test accuracy": test["accuracy"],
            "Test precision (macro)": test["precision_macro"],
            "Test recall (macro)": test["recall_macro"],
            "Test F1 (macro)": test["f1_macro"],
        }
    )
st.dataframe(
    pd.DataFrame(comparison_rows).style.format(
        {
            column: "{:.1%}"
            for column in [
                "CV macro F1", "Test accuracy", "Test precision (macro)",
                "Test recall (macro)", "Test F1 (macro)",
            ]
        }
    ),
    width="stretch",
    hide_index=True,
)

if st.session_state.feedback_entries:
    with st.expander(
        f"Corrections made this session ({len(st.session_state.feedback_entries)})",
        expanded=False,
    ):
        feedback_frame = pd.DataFrame(st.session_state.feedback_entries)
        st.dataframe(feedback_frame, width="stretch", hide_index=True)
        st.caption(
            "These labels are queued in this session only. Export them for review; "
            "they do not change model files or evaluation metrics."
        )
        feedback_csv = sanitize_csv_frame(feedback_frame).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download correction CSV",
            data=feedback_csv,
            file_name="review_corrections.csv",
            mime="text/csv",
            key="download_feedback",
        )

st.caption(
    "The dataset snapshot shows the two supplied sources separately. The deployed "
    "model trains only on the CG/OR training split; Amazon remains an external check. "
    "Scores do not guarantee detection of every "
    "kind of deceptive review or newer writing style."
)
