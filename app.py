# STEP 1 → Import libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# STEP 2 → Create app
app = Flask(__name__)
CORS(app)


# STEP 3 → Create dataset
data = {
    "email": [
        "Win money now",
        "Limited offer click now",
        "Hello friend how are you",
        "Meeting at 5 pm",
        "Congratulations you won prize",
        "Let's study together"
    ],
    "label": [1, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)


# STEP 4 → Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["email"])


# STEP 5 → Train model
model = MultinomialNB()
model.fit(X, df["label"])


# STEP 6 → Home route
@app.route("/")
def home():
    return "Backend Running ✅"


# STEP 7 → Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "email" not in data:
        return jsonify({"error": "Please send email text"}), 400

    email_text = data["email"]

    # Convert input text
    transformed = vectorizer.transform([email_text])

    # Predict
    prediction = model.predict(transformed)[0]

    result = "Spam" if prediction == 1 else "Not Spam"

    return jsonify({
        "email": email_text,
        "prediction": result
    })


# STEP 8 → Run server
if __name__ == "__main__":
    app.run(debug=True)
