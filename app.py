from flask import Flask, render_template, request
import joblib
from pathlib import Path
import pandas as pd

app = Flask(__name__)

# Project Paths

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR/"models"/"bank_marketing_rf_model.pkl"
DATA_PATH = BASE_DIR/'data'/'bank-full.csv'

df = pd.read_csv(DATA_PATH, sep=';')
print('Dataset loaded successfully', df.shape)

# Calculate campaign statistics

total_customers = len(df)

subscribed_customers = int((df["y"] == "yes").sum())

not_subscribed_customers = int((df["y"] == "no").sum())

subscription_rate = round(
(subscribed_customers / total_customers) * 100, 2
)

print("Total customers:", total_customers)
print("Subscribed customers:", subscribed_customers)
print("Not subscribed customers:", not_subscribed_customers)
print("Subscription rate:", subscription_rate, "%")


# Load the trained machine learning model
model = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = ["age",
                   "job",
                   "marital",
                   "education",
                   "default",
                   "balance",
                   "housing",
                   "loan",
                   "contact",
                   "day",
                   "month",
                   "campaign",
                   "pdays",
                   "previous",
                   "poutcome"]

@app.route("/")
def home():
    return render_template("index.html",
                           total_customers=total_customers,
                           subscribed_customers=subscribed_customers,
                           not_subscribed_customers=not_subscribed_customers,
                           subscription_rate=subscription_rate)

@app.route("/predict", methods=["GET"])
def prediction_form():
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        customer_data={"age": int(request.form["age"]),
                       "job": request.form["job"],
                       "marital": request.form["marital"],
                       "education": request.form["education"],
                       "default": request.form["default"],
                       "balance": int(request.form["balance"]),
                       "housing": request.form["housing"],
                       "loan": request.form["loan"],
                       "contact": request.form["contact"],
                       "day": int(request.form["day"]),
                       "month": request.form["month"],
                       "campaign": int(request.form["campaign"]),
                       "pdays": int(request.form["pdays"]),
                       "previous": int(request.form["previous"]),
                       "poutcome": request.form["poutcome"],}

        input_df = pd.DataFrame([customer_data], columns=FEATURE_COLUMNS)

        prediction = model.predict(input_df)[0]

        probabilities = model.predict_proba(input_df)[0]
        class_labels = list(model.classes_)
        yes_index = class_labels.index("yes")
        yes_probability = float(probabilities[yes_index])*100

        return render_template("result.html",
                               prediction = prediction,
                               probability = round(yes_probability, 2))
    
    except (KeyError, ValueError) as error:
        return f"Invalid or missing form input: {error}", 400

    except Exception:
        app.logger.exception("Prediction Failed")
        return "Prediction failed. Please check the server terminal for details.", 500

if __name__ == "__main__":
    app.run(debug=False)
