"""Train an exercise classifier from the supplied joint-angle CSV dataset.

Usage:
    python train_exercise_classifier.py
    python train_exercise_classifier.py --data path\\to\\exercise_angles.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


FEATURE_COLUMNS = ["Side", "Shoulder_Angle", "Elbow_Angle", "Hip_Angle", "Knee_Angle"]
TARGET_COLUMN = "Label"
LABEL_MAP = {"Pull ups": "curl", "Push Ups": "press", "Squats": "squat"}


def train(data_path: Path, output_path: Path) -> dict[str, object]:
    with data_path.open(newline="", encoding="utf-8-sig") as data_file:
        rows = list(csv.DictReader(data_file))
    if not rows:
        raise ValueError("Dataset contains no rows.")
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]).difference(rows[0])
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")

    # The supplied angle dataset contains several exercises, but the live
    # counter supports these three actions.  Train only on the compatible
    # labels and keep the mapping explicit in the fitted artifact.
    compatible_rows = [row for row in rows if row[TARGET_COLUMN].strip() in LABEL_MAP]
    features = [
        {"Side": "average", **{column: float(row[column]) for column in FEATURE_COLUMNS[1:]}}
        for row in compatible_rows
    ]
    target = [LABEL_MAP[row[TARGET_COLUMN].strip()] for row in compatible_rows]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    model = Pipeline(
        [
            ("vectorize", DictVectorizer(sparse=False)),
                ("classifier", DecisionTreeClassifier(max_depth=6, min_samples_leaf=8, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    accuracy = accuracy_score(y_test, predicted)
    report = classification_report(y_test, predicted, output_dict=True, zero_division=0)
    artifact = {
        "model": model,
        "features": FEATURE_COLUMNS,
        "labels": sorted(set(target)),
        "model_type": "decision_tree",
        "label_map": LABEL_MAP,
    }
    joblib.dump(artifact, output_path)

    metrics: dict[str, object] = {
        "rows": len(compatible_rows),
        "labels": artifact["labels"],
        "test_accuracy": round(float(accuracy), 4),
        "classification_report": report,
        "note": "This is a stratified sample split. Sequential samples from the same recording may be correlated, so validate on separate people/videos before relying on real-world accuracy.",
    }
    output_path.with_name("exercise_classifier_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the exercise angle classifier.")
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("exercise_angles.csv"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("exercise_classifier.joblib"))
    arguments = parser.parse_args()
    result = train(arguments.data, arguments.output)
    print(f"Trained on {result['rows']} rows; validation accuracy: {result['test_accuracy']:.2%}")
    print(f"Saved model to {arguments.output}")
