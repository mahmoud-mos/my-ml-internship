from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml_utils import (
    MODEL_CATEGORICAL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    OUTPUT_DIR,
    PROCESSED_DIR,
    display_path,
    precision_at_k,
    write_json,
)


FEATURE_PATH = PROCESSED_DIR / "refresh_feature_vector.csv"
BASELINE_PATH = PROCESSED_DIR / "baseline_refresh_queue.csv"
PREDICTION_PATH = PROCESSED_DIR / "model_predictions.csv"
RESULT_PATH = OUTPUT_DIR / "model_results.json"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train refresh opportunity models.")
    parser.add_argument("--features", default=str(FEATURE_PATH))
    parser.add_argument("--baseline", default=str(BASELINE_PATH))
    parser.add_argument("--predictions", default=str(PREDICTION_PATH))
    parser.add_argument("--results", default=str(RESULT_PATH))
    return parser.parse_args()


def build_feature_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    numeric_features = [
        column for column in MODEL_NUMERIC_FEATURES if column in frame.columns
    ]
    categorical_features = [
        column for column in MODEL_CATEGORICAL_FEATURES if column in frame.columns
    ]

    numeric_frame = frame[numeric_features].apply(pd.to_numeric, errors="coerce")
    numeric_frame = numeric_frame.replace([np.inf, -np.inf], np.nan).fillna(0)

    categorical_frame = frame[categorical_features].fillna("unknown").astype(str)
    encoded_frame = pd.get_dummies(
        categorical_frame,
        prefix=categorical_features,
        dummy_na=False,
        dtype=float,
    )

    feature_frame = pd.concat(
        [numeric_frame.reset_index(drop=True), encoded_frame.reset_index(drop=True)],
        axis=1,
    )
    return feature_frame, list(feature_frame.columns)


def perform_splits(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    # Time-forward holdout (last 20% by days_since_last_update)
    sorted_indices = frame.sort_values("days_since_last_update").index.to_numpy()
    holdout_size = int(len(frame) * 0.2)
    holdout_indices = sorted_indices[-holdout_size:]
    train_val_indices = sorted_indices[:-holdout_size]

    # 5-fold GroupKFold on the remainder
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(
        frame.iloc[train_val_indices], 
        frame.iloc[train_val_indices]["is_declining_label"], 
        groups=frame.iloc[train_val_indices]["client_id"]
    ))
    
    # Map folds back to original indices
    cv_folds = []
    for train_idx, val_idx in folds:
        cv_folds.append((train_val_indices[train_idx], train_val_indices[val_idx]))

    return train_val_indices, holdout_indices, cv_folds


def build_models() -> dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=5,
            min_samples_leaf=50,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced_subsample",
            max_depth=10,
            min_samples_leaf=25,
            n_estimators=200,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def predict_probability(model: object, feature_frame: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Model does not expose predict_proba: {type(model)!r}")
    probabilities = model.predict_proba(feature_frame)
    return np.asarray(probabilities[:, 1], dtype=float)


def metric_payload(
    target_series: pd.Series,
    probability_scores: np.ndarray,
    *,
    prefix: str = "",
) -> dict[str, float]:
    binary_predictions = (probability_scores >= 0.5).astype(int)
    payload = {
        f"{prefix}accuracy": float(accuracy_score(target_series, binary_predictions)),
        f"{prefix}precision": float(
            precision_score(target_series, binary_predictions, zero_division=0)
        ),
        f"{prefix}recall": float(
            recall_score(target_series, binary_predictions, zero_division=0)
        ),
        f"{prefix}f1": float(f1_score(target_series, binary_predictions, zero_division=0)),
        f"{prefix}precision_at_20": precision_at_k(target_series, probability_scores, 20),
        f"{prefix}precision_at_50": precision_at_k(target_series, probability_scores, 50),
        f"{prefix}precision_at_100": precision_at_k(target_series, probability_scores, 100),
    }
    if target_series.nunique() == 2:
        payload[f"{prefix}roc_auc"] = float(roc_auc_score(target_series, probability_scores))
        payload[f"{prefix}average_precision"] = float(
            average_precision_score(target_series, probability_scores)
        )
    else:
        payload[f"{prefix}roc_auc"] = 0.0
        payload[f"{prefix}average_precision"] = 0.0
    return payload


def top_feature_importance(
    model: object,
    feature_columns: list[str],
    *,
    limit: int = 25,
) -> list[dict[str, float | str]]:
    if isinstance(model, Pipeline):
        classifier = model.named_steps["model"]
        raw_values = np.abs(classifier.coef_[0])
    elif hasattr(model, "feature_importances_"):
        raw_values = np.asarray(model.feature_importances_, dtype=float)
    else:
        raw_values = np.zeros(len(feature_columns), dtype=float)

    importance_frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": raw_values,
        }
    )
    importance_frame = importance_frame.sort_values("importance", ascending=False).head(limit)
    return [
        {"feature": str(row.feature), "importance": float(row.importance)}
        for row in importance_frame.itertuples(index=False)
    ]


def main() -> None:
    args = parse_args()

    frame = pd.read_csv(args.features)
    baseline_frame = pd.read_csv(args.baseline)
    if frame.empty:
        raise ValueError("Feature vector is empty")
    if frame["is_declining_label"].nunique() < 2:
        raise ValueError("Training target has only one class; cannot train classifier")

    feature_frame, feature_columns = build_feature_matrix(frame)
    target_series = frame["is_declining_label"].astype(int)
    
    # Use new split logic
    train_val_indices, holdout_indices, cv_folds = perform_splits(frame)

    # Cross-validation
    cv_model_results: dict[str, list[dict[str, float]]] = {name: [] for name in build_models()}
    
    for train_idx, val_idx in cv_folds:
        train_features = feature_frame.iloc[train_idx]
        val_features = feature_frame.iloc[val_idx]
        train_target = target_series.iloc[train_idx]
        val_target = target_series.iloc[val_idx]
        
        trained_models = build_models()
        for model_name, model in trained_models.items():
            model.fit(train_features, train_target)
            val_probabilities = predict_probability(model, val_features)
            cv_model_results[model_name].append(metric_payload(val_target, val_probabilities))

    # Average CV results
    model_results = {}
    for model_name, results in cv_model_results.items():
        avg_metrics = {k: np.mean([r[k] for r in results]) for k in results[0]}
        model_results[model_name] = avg_metrics

    # Best model selection based on Precision@50
    best_model_name = sorted(
        model_results,
        key=lambda name: model_results[name]["precision_at_50"],
        reverse=True,
    )[0]

    # Final holdout evaluation
    holdout_features = feature_frame.iloc[holdout_indices]
    holdout_target = target_series.iloc[holdout_indices]
    
    full_data_models = build_models()
    best_full_model = full_data_models[best_model_name]
    best_full_model.fit(feature_frame.iloc[train_val_indices], target_series.iloc[train_val_indices])
    holdout_probabilities = predict_probability(best_full_model, holdout_features)
    
    holdout_metrics = metric_payload(holdout_target, holdout_probabilities, prefix="holdout_")
    
    # Baseline comparison (for holdout)
    baseline_lookup = baseline_frame.set_index("content_id")["baseline_refresh_score"]
    baseline_holdout_scores = (
        frame.iloc[holdout_indices]["content_id"].map(baseline_lookup).fillna(0).to_numpy()
    )
    baseline_metrics = metric_payload(holdout_target, baseline_holdout_scores, prefix="baseline_")

    print(f"Cross-Validation Best Model (Precision@50): {best_model_name}")
    print(f"Holdout Precision@50: {holdout_metrics['holdout_precision_at_50']:.4f}")
    print(f"Baseline Precision@50: {baseline_metrics['baseline_precision_at_50']:.4f}")

    # Predictions for full data
    prediction_frame = frame[["content_id", "client_id", "is_declining_label"]].copy()
    
    # Re-train on full data
    best_model_final = build_models()[best_model_name]
    best_model_final.fit(feature_frame, target_series)
    prediction_frame[f"prob_{best_model_name}"] = predict_probability(best_model_final, feature_frame)
    prediction_frame["best_model_name"] = best_model_name
    
    prediction_path = Path(args.predictions)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(prediction_path, index=False)

    results_payload = {
        "input_rows": int(len(frame)),
        "train_val_rows": int(len(train_val_indices)),
        "holdout_rows": int(len(holdout_indices)),
        "target": "is_declining_label",
        "models_cv": model_results,
        "holdout": holdout_metrics,
        "baseline_holdout": baseline_metrics,
        "best_model": {
            "name": best_model_name,
            "selection_metric": "precision_at_50",
            "feature_importance_top": top_feature_importance(best_full_model, feature_columns),
        },
    }
    write_json(Path(args.results), results_payload)


if __name__ == "__main__":
    main()
