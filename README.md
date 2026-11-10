# Explainable and Drift-Aware MLOps Pipeline for Heart Disease Prediction

## Overview

This project implements an explainable and drift-aware Machine Learning Operations (MLOps) pipeline for heart disease prediction. It combines interpretable machine learning with experiment tracking, cloud deployment, and data drift monitoring.

The system uses an Explainable Boosting Machine (EBM) as its primary prediction model and Logistic Regression as a baseline model. It integrates MLflow for experiment tracking, Evidently AI for data drift monitoring, and AWS services for deployment, artifact storage, logging, and alert notifications.

## Key Features

- Heart disease prediction using machine learning.
- Explainable AI using global and local feature importance.
- Model experiment tracking using MLflow.
- Data drift monitoring using Evidently AI.
- REST API deployment using FastAPI.
- AWS-based infrastructure for deployment and monitoring.
- Prediction artifact storage using Amazon S3.
- Logging and drift alert notifications using Amazon CloudWatch and Amazon SNS.

## Architecture

The project follows these major stages:

1. **Dataset Preparation:** Load, clean, and preprocess the heart disease dataset.
2. **Model Development:** Train and evaluate EBM and Logistic Regression models.
3. **Experiment Tracking:** Track model experiments and evaluation metrics using MLflow.
4. **Model Explainability:** Use EBM to understand global feature importance and individual predictions.
5. **Model Deployment:** Deploy the prediction application using FastAPI on AWS EC2.
6. **Artifact Storage:** Use Amazon S3 for storing prediction-related artifacts and monitoring reports.
7. **Drift Monitoring:** Use Evidently AI to detect changes in incoming prediction data.
8. **Logging and Alerts:** Use Amazon CloudWatch for logging and Amazon SNS for drift alert notifications.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Explainable Boosting Machine (EBM), Logistic Regression |
| Explainability | InterpretML |
| Experiment Tracking | MLflow |
| Data Processing | Pandas, NumPy |
| Drift Monitoring | Evidently AI |
| Backend | FastAPI |
| Cloud Infrastructure | AWS EC2, Amazon S3 |
| Monitoring and Alerts | Amazon CloudWatch, Amazon SNS |
| Frontend | HTML, CSS, JavaScript, Bootstrap |
| Visualization | Chart.js |
| Model Serialization | Skops |

## Dataset

- **Dataset:** UCI Statlog Heart Disease Dataset
- **Number of Records:** 270
- **Number of Features:** 13
- **Target:** Binary classification (heart disease presence or absence)

The dataset is used for preprocessing, model training, evaluation, and reference data for drift monitoring.

## Model Performance

The Explainable Boosting Machine achieved approximately **87.04% test accuracy** during evaluation.

Logistic Regression was also implemented as a baseline model for comparison.

## Project Structure

```text
heart-disease-mlops/
│
├── app.py
├── drift_monitor.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── data/
│   └── reference_data.csv
│
├── notebooks/
│   └── model_development.ipynb
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
└── templates/
    └── index.html
```

The trained model artifact is generated during the model training stage and stored separately under `models/verified_ebm/`.

## Installation and Setup

Follow these steps to set up and run the project locally.

### 1. Clone the repository

```bash
git clone https://github.com/shreesm17/Explainable-drift-aware-heart-disease-prediction.git

cd Explainable-drift-aware-heart-disease-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment.

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Model Training and Artifact Generation

### 4. Prepare the dataset

- Use the reference dataset provided in the `data/` directory.
- Open the model development notebook:

```text
notebooks/model_development.ipynb
```

- Execute the notebook cells to perform data loading, preprocessing, feature preparation, and train-test splitting.

### 5. Train and evaluate the models

- Train the Explainable Boosting Machine (EBM) model.
- Train Logistic Regression as the baseline model.
- Evaluate both models using appropriate classification metrics.
- Review the model performance and explainability outputs.

### 6. Save the trained model artifact

After training and evaluating the EBM model:

- Serialize the trained model using the Skops library.
- Save the model artifact as `model.skops`.
- Store it in the following directory:

```text
models/
└── verified_ebm/
    └── model.skops
```

Create the directories if they do not already exist.

The application uses this trained model artifact for prediction functionality.

## Running the Application

### 7. Configure environment variables

Use `.env.example` as a reference to identify the required AWS configuration values.

Configure the required environment variables in your local environment or deployment environment, including:

- AWS region
- S3 bucket name
- Drift report storage location
- SNS topic ARN

Ensure that the AWS credentials and permissions required by the application are configured.

### 8. Start the FastAPI application

Run:

```bash
uvicorn app:app --reload
```

Open the application in your browser:

```text
http://127.0.0.1:8000
```

## Data Drift Monitoring

### 9. Run the drift monitoring script

Execute:

```bash
python drift_monitor.py
```

The monitoring script uses Evidently AI to compare reference data with incoming prediction data and identify potential data drift.

Monitoring reports can be stored using the configured AWS S3 settings, while logging and alert notifications use the configured CloudWatch and SNS services.

## AWS Deployment

The application is deployed using Amazon EC2.

The AWS infrastructure supports:

- **Amazon EC2:** Hosting the FastAPI application.
- **Amazon S3:** Storing prediction-related artifacts and monitoring reports.
- **Amazon CloudWatch:** Application and monitoring logs.
- **Amazon SNS:** Drift alert notifications.

## Future Enhancements

- Automated model retraining when significant drift is detected.
- Continuous model performance monitoring.
- Enhanced prediction and monitoring dashboards.
- Integration of additional explainability techniques.
