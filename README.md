Fake News Detection Web Application

An AI-powered Fake News Detection Web App built using Natural Language Processing (NLP) and Machine Learning.
The system classifies news articles as Fake or Real using TF-IDF vectorization and Logistic Regression.

🚀 Features

Detects Fake vs Real News

NLP-based text preprocessing

TF-IDF Feature Extraction

Logistic Regression Classifier

Confidence score prediction

Flask-based Web Interface

Model persistence using Pickle

Ready for Cloud Deployment (Render compatible)

🧠 Tech Stack

Python

Flask

Scikit-learn

Pandas

TF-IDF (NLP)

Logistic Regression

HTML/CSS

Gunicorn (for deployment)

📂 Project Structure
fake_news_app/
│
├── app.py
├── train_model.py
├── requirements.txt
│
├── dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── model/
│   ├── fake_news_model.pkl
│   └── vectorizer.pkl
│
├── templates/
│   ├── base.html
│   ├── index.html
│   └── result.html
│
└── static/
    └── style.css
⚙️ How It Works

News dataset is loaded (Fake & True news).

Text is converted into numerical features using TF-IDF Vectorizer.

A Logistic Regression model is trained on the dataset.

Model and vectorizer are saved using Pickle.

User enters news text via web interface.

Model predicts whether the news is Fake or Real.

Confidence score is displayed.
