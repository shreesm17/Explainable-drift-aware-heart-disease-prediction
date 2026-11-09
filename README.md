
# Explainable and Drift-Aware MLOps Pipeline for Heart Disease Prediction

## Overview

This project implements an explainable and drift-aware machine learning pipeline for heart disease prediction. It combines interpretable machine learning with experiment tracking, cloud deployment, and data drift monitoring.

The system uses an Explainable Boosting Machine (EBM) as its primary prediction model and Logistic Regression as a baseline model.

## Key Features

- Heart disease prediction using machine learning.
- Explainable AI using global and local feature importance.
- Model experiment tracking using MLflow.
- Data drift monitoring using Evidently AI.
- REST API deployment using FastAPI.
- AWS-based infrastructure for deployment and monitoring.
- Drift alert notifications using Amazon SNS.

## Architecture

1. Dataset preparation and preprocessing.
2. Model training and evaluation.
3. Experiment tracking using MLflow.
4. Model explainability using EBM.
5. Model deployment through FastAPI on AWS EC2.
6. Prediction artifact storage using Amazon S3.
7. Data drift monitoring using Evidently AI.
8. Logging and drift notifications using Amazon CloudWatch and SNS.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Machine Learning | EBM, Logistic Regression |
| Explainability | InterpretML |
| Experiment Tracking | MLflow |
| Data Processing | Pandas, NumPy |
| Drift Monitoring | Evidently AI |
| Backend | FastAPI |
| Cloud | AWS EC2, S3, CloudWatch, SNS |
| Frontend | HTML, CSS, JavaScript, Bootstrap |
| Visualization | Chart.js |

## Model Performance

The Explainable Boosting Machine achieved approximately **87.04% test accuracy** during evaluation.

Logistic Regression was also implemented as a baseline for comparison.

## Project Structure

```text
heart-disease-mlops/
├── app.py
├── drift_monitor.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/
│   └── reference_data.csv
├── models/
│   └── verified_ebm/
│       └── model.skops
├── notebooks/
│   └── model_development.ipynb
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── templates/
│   └── index.html
└── screenshots/
```

## Running the Project Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd heart-disease-mlops
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and configure the required AWS settings.

### 5. Start the application

```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

**Note:** AWS-backed drift report storage and SNS notifications require valid AWS configuration and permissions.


