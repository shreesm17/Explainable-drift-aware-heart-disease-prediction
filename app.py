
import time
import logging
from drift_monitor import run_drift_analysis
import numpy as np
import boto3
import csv
import os
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import skops.io as sio

# AWS configuration
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
S3_BUCKET = os.getenv("S3_BUCKET")
DRIFT_REPORT_KEY = os.getenv(
    "DRIFT_REPORT_KEY",
    "artifacts/monitoring/drift_monitoring_results.json"
)
SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN")

sns_client = boto3.client("sns", region_name=AWS_REGION)
s3_client = boto3.client("s3", region_name=AWS_REGION)

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Explainable and Drift-Aware Heart Disease Prediction",
    version="2.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Application logging
APP_LOG_DIR = "logs"
os.makedirs(APP_LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(APP_LOG_DIR, "app.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("heart_disease_mlops")
logger.info("Heart Disease Prediction application initialized")

# Load verified EBM model
MODEL_PATH = "models/verified_ebm/model.skops"

trusted_types = [
    "interpret.glassbox._ebm._ebm.ExplainableBoostingClassifier"
]

model = sio.load(MODEL_PATH, trusted=trusted_types)


# Patient input schema
class PatientData(BaseModel):
    age: float
    sex: int
    chest_pain: int
    rest_bp: float
    serum_chol: float
    fasting_blood_sugar: int
    electrocardiographic: int
    max_heart_rate: float
    angina: int
    oldpeak: float
    slope: int
    major_vessels: int
    thal: int


FEATURE_NAMES = [
    "age", "sex", "chest_pain", "rest_bp", "serum_chol",
    "fasting_blood_sugar", "electrocardiographic",
    "max_heart_rate", "angina", "oldpeak", "slope",
    "major_vessels", "thal"
]


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/explain/global")
def explain_global():
    explanation = model.explain_global()
    data = explanation.data()

    importance = sorted(
        zip(data["names"], data["scores"]),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return {
        "explanation_type": "global",
        "feature_importance": [
            {
                "feature": str(name),
                "importance": float(score)
            }
            for name, score in importance
        ]
    }


# Prediction logging
LOG_DIR = "data"
LOG_FILE = os.path.join(LOG_DIR, "prediction_logs.csv")


@app.post("/predict")
def predict(data: PatientData):
    features = np.array([[
        data.age,
        data.sex,
        data.chest_pain,
        data.rest_bp,
        data.serum_chol,
        data.fasting_blood_sugar,
        data.electrocardiographic,
        data.max_heart_rate,
        data.angina,
        data.oldpeak,
        data.slope,
        data.major_vessels,
        data.thal
    ]])

    inference_start = time.perf_counter()
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    inference_time_ms = (
        time.perf_counter() - inference_start
    ) * 1000

    # Individual explanation
    explanation = model.explain_local(features, [prediction])
    local_data = explanation.data(0)

    contributions = [
        {
            "feature": str(name),
            "value": str(value),
            "contribution": float(score)
        }
        for name, value, score in zip(
            local_data["names"],
            local_data["values"],
            local_data["scores"]
        )
    ]

    contributions.sort(
        key=lambda x: abs(x["contribution"]),
        reverse=True
    )

    # Store prediction record locally
    os.makedirs(LOG_DIR, exist_ok=True)

    log_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **dict(zip(FEATURE_NAMES, features[0].tolist())),
        "prediction": prediction,
        "class_0_probability": float(probabilities[0]),
        "class_1_probability": float(probabilities[1])
    }

    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=log_record.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(log_record)

    return {
        "prediction": prediction,
        "class_0_probability": round(float(probabilities[0]), 4),
        "class_1_probability": round(float(probabilities[1]), 4),
        "model": "ExplainableBoostingClassifier",
        "inference_time_ms": round(inference_time_ms, 4),
        "explanation": {
            "type": "local",
            "feature_contributions": contributions
        }
    }


@app.get("/monitor/drift")
def monitor_drift():
    try:
        result = run_drift_analysis()

        if result.get("status") == "DRIFT_ALERT":

            if SNS_TOPIC_ARN:
                subject = "Heart Disease MLOps - Data Drift Alert"

                message = f"""
Data drift has been detected in the Heart Disease Prediction application.

Project: Heart Disease Prediction MLOps
Status: {result.get('status')}
Drifted Features: {result.get('drifted_features')}
Total Features: {result.get('total_features')}
Drift Percentage: {result.get('drift_share', 0) * 100:.2f}%
Alert Threshold: {result.get('threshold', 0.20) * 100:.0f}%
Prediction Count: {result.get('prediction_count')}

Please review the latest Evidently AI drift monitoring report.
"""

                sns_client.publish(
                    TopicArn=SNS_TOPIC_ARN,
                    Subject=subject,
                    Message=message
                )

                logger.warning("SNS drift notification published")

            else:
                logger.warning(
                    "Drift detected, but SNS_TOPIC_ARN is not configured"
                )

        else:
            logger.info(
                "Drift analysis completed with status: %s",
                result.get("status")
            )

        return result

    except Exception:
        logger.exception("Drift monitoring or SNS notification failed")

        return {
            "status": "ERROR",
            "message": "Drift monitoring or notification failed."
        }
