from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    data = vectorizer.transform([news])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0].max() * 100

    if prediction == 0:
        result = "Fake News ❌"
    else:
        result = "Real News ✅"

    return render_template("result.html", result=result, probability=round(probability,2))

if __name__ == "__main__":
    app.run(debug=True)