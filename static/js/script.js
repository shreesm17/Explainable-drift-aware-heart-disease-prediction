const fields = [
    ["age", "Age", 57],
    ["sex", "Sex (0 = Female, 1 = Male)", 1],
    ["chest_pain", "Chest Pain Type (0–3)", 2],
    ["rest_bp", "Resting Blood Pressure", 140],
    ["serum_chol", "Serum Cholesterol", 240],
    ["fasting_blood_sugar", "Fasting Blood Sugar (0 or 1)", 0],
    ["electrocardiographic", "Electrocardiographic Results (0–2)", 0],
    ["max_heart_rate", "Maximum Heart Rate", 150],
    ["angina", "Exercise-Induced Angina (0 or 1)", 0],
    ["oldpeak", "Oldpeak", 0.8],
    ["slope", "Slope (0–2)", 1],
    ["major_vessels", "Major Vessels (0–3)", 0],
    ["thal", "Thal (0–3)", 3]
];

const formFields = document.getElementById("formFields");

fields.forEach(([name, label, value]) => {
    formFields.innerHTML += `
        <div class="col-md-6 mb-3">
            <label class="form-label">${label}</label>
            <input
                type="number"
                class="form-control"
                name="${name}"
                value="${value}"
                step="any"
                required
            >
        </div>
    `;
});

let localChart = null;
let globalChart = null;

// Patient prediction
document.getElementById("predictionForm").addEventListener("submit", async (event) => {
    event.preventDefault();

    const formData = new FormData(event.target);
    const patientData = {};

    fields.forEach(([name]) => {
        patientData[name] = Number(formData.get(name));
    });

    const resultDiv = document.getElementById("result");
    resultDiv.innerHTML = "Processing prediction...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(patientData)
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(JSON.stringify(result));
        }

        const predictionText =
            result.prediction === 0
                ? "Class 0 (Negative)"
                : "Class 1 (Positive)";

        resultDiv.innerHTML = `
            <div class="alert alert-info">
                <h5>Prediction Result</h5>
                <p><strong>Prediction:</strong> ${predictionText}</p>
                <p><strong>Class 0 Probability:</strong>
                    ${(result.class_0_probability * 100).toFixed(2)}%</p>
                <p><strong>Class 1 Probability:</strong>
                    ${(result.class_1_probability * 100).toFixed(2)}%</p>
                <p><strong>Model:</strong> ${result.model}</p>
                <p><strong>Inference Time:</strong>
                    ${Number(result.inference_time_ms).toFixed(4)} ms</p>
            </div>
            
        `;
         await loadDriftReport();

        // Individual explanation chart
        const contributions = result.explanation.feature_contributions;

        const labels = contributions.map(item => item.feature);
        const scores = contributions.map(item => item.contribution);

        if (localChart) {
            localChart.destroy();
        }

        localChart = new Chart(document.getElementById("localChart"), {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "Feature Contribution",
                    data: scores,
                    backgroundColor: scores.map(score =>
                        score >= 0 ? "#dc6b6b" : "#4c9f70"
                    )
                }]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: "Contribution to model score"
                        }
                    }
                }
            }
        });

    } catch (error) {
        resultDiv.innerHTML = `
            <div class="alert alert-danger">
                Error: ${error.message}
            </div>
        `;
    }
});


// Global feature importance
document.getElementById("loadGlobal").addEventListener("click", async () => {
    const button = document.getElementById("loadGlobal");
    button.textContent = "Loading...";

    try {
        const response = await fetch("/explain/global");
        const result = await response.json();

        if (!response.ok) {
            throw new Error(JSON.stringify(result));
        }

        const importance = result.feature_importance;

        if (globalChart) {
            globalChart.destroy();
        }

        globalChart = new Chart(document.getElementById("globalChart"), {
            type: "bar",
            data: {
                labels: importance.map(item => item.feature),
                datasets: [{
                    label: "Global Importance",
                    data: importance.map(item => item.importance),
                    backgroundColor: "#2878b5"
                }]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: "Feature Importance Score"
                        }
                    }
                }
            }
        });

        button.textContent = "Refresh Global Feature Importance";

    } catch (error) {
        button.textContent = "Load Global Feature Importance";
        alert("Unable to load global explanation: " + error.message);
    }
});


// Load dynamic drift monitoring results
async function loadDriftReport() {
    const errorElement = document.getElementById("driftError");
    const statusElement = document.getElementById("driftStatus");

    errorElement.textContent = "Running drift analysis...";

    try {
        const response = await fetch("/monitor/drift");

        if (!response.ok) {
            throw new Error("Failed to run drift analysis");
        }

        const data = await response.json();
        // Update last analysis time
        const timestampElement = document.getElementById("driftTimestamp");

   if (timestampElement) {
    timestampElement.textContent = new Date().toLocaleString();
   }

        statusElement.textContent = data.status;

        statusElement.className =
            data.status === "DRIFT_ALERT"
                ? "badge bg-danger"
                : data.status === "NORMAL"
                ? "badge bg-success"
                : "badge bg-secondary";

        // Prediction count
        const countElement = document.getElementById("predictionCount");
        if (countElement) {
            countElement.textContent =
                `${data.prediction_count} predictions`;
        }

        // Drift percentage
        const percentageElement =
            document.getElementById("driftPercentage");

        if (percentageElement) {
            percentageElement.textContent =
                data.drift_share !== undefined
                    ? (data.drift_share * 100).toFixed(2) + "%"
                    : "Not available";
        }

        // Alert threshold
        const thresholdElement =
            document.getElementById("driftThreshold");

        if (thresholdElement) {
            thresholdElement.textContent =
                data.threshold !== undefined
                    ? (data.threshold * 100).toFixed(2) + "%"
                    : "20.00%";
        }

        // Monitoring tool
        const toolElement = document.getElementById("monitoringTool");
        if (toolElement) {
            toolElement.textContent = "Evidently AI";
        }

        // Number of drifted features
        const featureElement =
            document.getElementById("driftedFeatures");

        if (featureElement) {
            featureElement.textContent =
                data.drifted_features !== undefined
                    ? `${data.drifted_features}/${data.total_features}`
                    : "Not calculated";
        }

        errorElement.textContent =
            data.message || "Drift analysis completed.";

    } catch (error) {
        statusElement.textContent = "Unavailable";
        statusElement.className = "badge bg-secondary";
        errorElement.textContent = error.message;
    }
}