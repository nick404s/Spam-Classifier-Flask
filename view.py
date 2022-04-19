from flask import Flask, render_template, request
import os
import numpy as np
from sklearn.pipeline import Pipeline
import joblib


app = Flask(__name__)

model = joblib.load('email_classifer.pkl')


@app.route("/")
@app.route("/home")
def home_page():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    result = ""
    if request.method == 'POST' and "text" in request.form:
        user_text = request.form.get("text")
        prediction = model.predict(np.array([user_text]))
        if prediction[0] == "ham":
            result = "not a spam"
        else:
            result = "spam"

    return render_template('index.html', prediction=f'The entered email is {result}')


@app.route("/model")
def model_page():
    items = [
        {'id': 1, 'parameter': 'Accuracy', 'value': '98.56%'},
        {'id': 2, 'parameter': 'Precision (ham/spam)', 'value': '0.99 / 0.98'},
        {'id': 3, 'parameter': 'Recall (ham/spam)', 'value': '1.00 / 0.91'},
        {'id': 4, 'parameter': 'F1-Score (ham/spam)', 'value': '0.99 / 094'}
    ]
    return render_template("model.html", items=items)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)