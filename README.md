# CampaignIQ — Bank Marketing Response Prediction

A machine learning web application that predicts whether a customer is likely to subscribe to a bank term deposit based on historical marketing campaign data.

## Overview

CampaignIQ combines data analysis, statistical techniques, and machine learning to explore customer response patterns and estimate subscription probability through an interactive Flask application.

## Key Features

* **Data Analysis:** Data preprocessing, exploratory data analysis (EDA), and statistical analysis.
* **Machine Learning:** Tuned Random Forest classifier for customer response prediction.
* **Interactive Web App:** Flask-based interface for entering customer details and viewing predictions.
* **Probability Estimates:** Displays the predicted outcome and estimated subscription probability.
* **Responsive UI:** Custom beige-and-espresso design with a homepage, prediction form, and results page.

## Model Performance

The tuned Random Forest model was evaluated on a held-out test dataset.

| Metric                   |  Score |
| ------------------------ | -----: |
| Accuracy                 | 86.01% |
| Precision (Subscription) |  42.5% |
| Recall (Subscription)    |  55.5% |
| F1-Score (Subscription)  |  48.1% |

*The dataset is imbalanced, so accuracy alone does not fully represent model performance. Precision, recall, and F1-score provide additional insight into the model's ability to identify potential subscribers.*

**Feature selection:** Call duration (`duration`) was excluded because it is unavailable before a marketing call. This keeps predictions focused on information available before contact.

## Tech Stack

* **Languages:** Python, SQL
* **Data Analysis:** Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn, Random Forest, GridSearchCV
* **Web Application:** Flask, HTML, CSS, Bootstrap
* **Model Persistence:** Joblib
* **Development:** Jupyter Notebook, VS Code, Git, GitHub

## Project Structure

```text
campaign-response-analysis/
├── app.py
├── requirements.txt
├── data/
│   └── bank-full.csv
├── models/
│   └── bank_marketing_rf_model.pkl
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_eda_analysis.ipynb
│   ├── 03_statistical_analysis.ipynb
│   └── 04_machine_learning.ipynb
├── templates/
│   ├── index.html
│   ├── predict.html
│   └── result.html
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── script.js
```

## Run Locally

**1. Clone the repository**

```bash
git clone <your-repository-url>
cd campaign-response-analysis
```

**2. Install dependencies**

```bash
python -m pip install -r requirements.txt
```

**3. Start the Flask application**

```bash
python app.py
```

**4. Open the application**

Visit `http://127.0.0.1:5000` in your browser.

Ensure the dataset and saved model are available at the paths expected by the application.

## Live Demo

🚀 **CampaignIQ is deployed and publicly accessible!**

Try the application here:

**[Launch CampaignIQ](https://campaign-response-analysis.onrender.com/)**

The application is hosted on Render and provides an interactive interface for predicting customer subscription likelihood using a trained Random Forest model.

### Deployment Details

* **Platform:** Render
* **Framework:** Flask
* **Production Server:** Gunicorn
* **Source Code:** GitHub
* **Deployment:** Automatic deployment from the `main` branch.

**Note:** The application uses Render's free hosting plan. The service may spin down after inactivity, so the first request can take longer to load.


## Dataset

The project uses the Bank Marketing dataset, containing customer information and historical campaign outcomes. The target variable, `y`, indicates whether a customer subscribed to a term deposit.

## Limitations

Predictions are estimates based on historical data, not guarantees of customer behavior. Model probabilities may not be perfectly calibrated, and results should be interpreted alongside other relevant information.

---


