from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return "Spam Email Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()
    email_text = data["email"]

    email_vector = vectorizer.transform([email_text])

    prediction = model.predict(email_vector)[0]

    if prediction == 1:
        result = "Spam"
    else:
        result = "Not Spam"

    return jsonify({"prediction": result})

app.run(debug=True)