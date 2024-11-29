from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Conversion rate (assumed USD to INR for simplicity; you can adjust or make it dynamic)
INR_TO_USD = 0.012  # 1 INR = 0.012 USD (for example)

# Load and preprocess the dataset
def prepare_data():
    # Load the dataset
    data = pd.read_csv("data/Credit_Score_Clean.csv")
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    data["Credit_Score"] = label_encoder.fit_transform(data["Credit_Score"])
    
    # Define features (X) and target (y)
    X = data[[
        "Annual_Income", "Num_of_Loan", "Credit_Utilization_Ratio",
        "Delay_from_due_date", "Num_of_Delayed_Payment", "Credit_History_Age_Months"
    ]]
    y = data["Credit_Score"]
    
    return X, y, label_encoder

# Train the model
def train_model():
    X, y, label_encoder = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, label_encoder

# Train the model on app startup
model, label_encoder = train_model()

# Route for home page
@app.route("/")
def index():
    return render_template("index.html")

# Route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the annual income in INR and convert to USD
        annual_income_inr = float(request.form["Annual_Income"])
        annual_income_usd = annual_income_inr * INR_TO_USD  # Convert INR to USD
        
        # Extract other features from the form
        features = [
            annual_income_usd,
            int(request.form["Num_of_Loan"]),
            float(request.form["Credit_Utilization_Ratio"]),
            int(request.form["Delay_from_due_date"]),
            int(request.form["Num_of_Delayed_Payment"]),
            int(request.form["Credit_History_Age_Months"]),
        ]
        
        # Prepare input data
        input_data = np.array([features])
        
        # Make prediction using the trained model
        prediction = model.predict(input_data)[0]
        
        # Map prediction to original label
        result = label_encoder.inverse_transform([prediction])[0]
        
        # Set color based on the prediction result
        if result == "Poor":
            color = "red"
        else:
            color = "green"
        
        return jsonify({
            "success": True,
            "credit_score": result,
            "color": color
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
