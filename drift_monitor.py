
import pandas as pd
from pathlib import Path
from evidently import Report
from evidently.presets import DataDriftPreset

BASE_DIR = Path(__file__).resolve().parent

REFERENCE_FILE = BASE_DIR / "data" / "reference_data.csv"
PREDICTION_FILE = BASE_DIR / "data" / "prediction_logs.csv"

MIN_RECORDS = 30
DRIFT_THRESHOLD = 0.20

FEATURE_NAMES = [
    "age", "sex", "chest-pain", "rest-bp", "serum-chol",
    "fasting-blood-sugar", "electrocardiographic",
    "max-heart-rate", "angina", "oldpeak", "slope",
    "major-vessels", "thal"
]


def run_drift_analysis():

    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Reference dataset not found: {REFERENCE_FILE}"
        )

    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Prediction log not found: {PREDICTION_FILE}"
        )

    reference = pd.read_csv(REFERENCE_FILE)
    reference = reference[FEATURE_NAMES]

    current = pd.read_csv(PREDICTION_FILE)

    if len(current) < MIN_RECORDS:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": f"Need at least {MIN_RECORDS} predictions.",
            "prediction_count": len(current),
            "required_count": MIN_RECORDS
        }

    rename_columns = {
        column: column.replace("_", "-")
        for column in current.columns
        if column in [
            "chest_pain", "rest_bp", "serum_chol",
            "fasting_blood_sugar", "max_heart_rate",
            "major_vessels"
        ]
    }

    current = current.rename(columns=rename_columns)
    current = current[FEATURE_NAMES]

    report = Report([
        DataDriftPreset()
    ])

    result = report.run(
        reference_data=reference,
        current_data=current
    )

    report_data = result.dict()

    drifted_features = 0

    for metric in report_data.get("metrics", []):
        if "DriftedColumnsCount" in str(metric.get("metric_id", "")):
            drifted_features = metric["value"]["count"]

    drift_share = drifted_features / len(FEATURE_NAMES)

    status = (
        "DRIFT_ALERT"
        if drift_share >= DRIFT_THRESHOLD
        else "NORMAL"
    )

    return {
        "status": status,
        "prediction_count": len(current),
        "drifted_features": drifted_features,
        "total_features": len(FEATURE_NAMES),
        "drift_share": round(drift_share, 4),
        "threshold": DRIFT_THRESHOLD
    }
