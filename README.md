# 🧠 ScriptMind AI
### Author Attribution & Stylometric Detection using NLP and Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-green)
![React](https://img.shields.io/badge/React-Frontend-61DAFB)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![NLP](https://img.shields.io/badge/NLP-Stylometry-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Overview

ScriptMind AI is an NLP-powered authorship attribution system that identifies the most probable author of a text passage by analyzing writing style, vocabulary usage, grammatical patterns, and linguistic fingerprints.

Unlike traditional text classification systems that rely only on word frequencies, ScriptMind AI combines:

- Stylometric Features
- Linguistic Fingerprinting
- TF-IDF N-Gram Analysis
- Machine Learning Classification

to create a hybrid author detection pipeline.

The system extracts unique writing signatures from literary text and predicts the author in real time through an interactive web interface.

---

## ✨ Features

### 📖 Author Attribution
Predicts the most likely author of a given text passage.

### 🧠 Stylometric Analysis
Extracts writing-style characteristics such as:

- Vocabulary Richness
- Lexical Diversity
- Sentence Complexity
- Stopword Usage
- Part-of-Speech Distribution
- Punctuation Patterns
- Capitalization Habits

### 🔍 Linguistic Fingerprinting

Uses TF-IDF based:

- Unigrams
- Bigrams

to capture author-specific language patterns.

### ⚡ Real-Time Prediction

Users can paste text directly into the interface and receive predictions instantly.

### 🌐 Full Stack Application

- React + Tailwind Frontend
- Flask Backend API
- Scikit-Learn Model Inference

---

# 📸 Screenshots

## Landing Page

A modern cyberpunk-inspired UI allowing users to start author detection.

---

## Text Analysis Interface

Paste literary text and run real-time author attribution.

---

## Prediction Result

Displays:

- Predicted Author
- Confidence Level
- Stylometric Indicators
- Machine Learning Classification Result

---

# 🏗️ System Architecture

```text
User Input Passage
        │
        ▼
Text Cleaning
(Remove HTML Tags)
        │
        ▼
Tokenization
(NLTK)
        │
        ▼
+----------------------------------+
| Stylometric Feature Extraction   |
+----------------------------------+
        │
        ├── Word Count
        ├── Character Count
        ├── Sentence Count
        ├── Avg Word Length
        ├── Avg Sentence Length
        ├── Type Token Ratio
        ├── POS Ratios
        ├── Stopword Ratio
        ├── Punctuation Usage
        └── Uppercase Frequency
        │
        ▼
TF-IDF Vectorization
(Unigrams + Bigrams)
        │
        ▼
Feature Fusion
(Hybrid Feature Space)
        │
        ▼
Standard Scaling
        │
        ▼
Random Forest Classifier
        │
        ▼
Predicted Author
        │
        ▼
React Frontend
```

---

# 📂 Project Structure

```text
STYLOMETRIC/
│
├── backend/
│   │
│   ├── app.py
│   ├── requirements.txt
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── vectorizer.pkl
│   ├── label_encoder.pkl
│   │
│   └── __pycache__/
│
├── frontend/
│   │
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── App.jsx
│   │   └── main.jsx
│   │
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── index.html
│
├── pipeline/
│   │
│   ├── auth_datasets/
│   │   └── authors_dataset.csv
│   │
│   ├── text_preprocessing.ipynb
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── vectorizer.pkl
│   └── label_encoder.pkl
│
└── README.md
```

---

# 🔬 Dataset

The model is trained on a curated collection of literary works from multiple authors.

Dataset contains:

- Author Name
- Literary Passage
- Cleaned Text Content

The dataset was preprocessed and transformed into stylometric and linguistic feature vectors before training.

---

# ⚙️ Feature Engineering

## 1. Stylometric Features

The system extracts handcrafted writing-style features including:

```python
{
    "char_count",
    "word_count",
    "sent_count",
    "avg_word_len",
    "avg_sentence_len",
    "unique_word_count",
    "type_token_ratio",
    "noun_ratio",
    "verb_ratio",
    "adj_ratio",
    "adv_ratio",
    "punctuation_count",
    "upper_case_count",
    "stopword_count",
    "stopword_ratio"
}
```

### Additional POS Features

For every detected Part-of-Speech tag:

```python
features[f'pos_{tag}']
```

is generated dynamically.

---

## 2. TF-IDF Vectorization

The project uses:

```python
TfidfVectorizer(
    ngram_range=(1,2),
    max_features=10000,
    min_df=2,
    max_df=0.9
)
```

This captures:

- Word Frequencies
- Word Pairs
- Author Vocabulary Preferences

---

## 3. Hybrid Feature Space

Stylometric Features:

```text
34+
```

Combined With:

```text
10,000 TF-IDF Features
```

Using:

```python
hstack([
    X_ngram,
    X_style_sparse
])
```

Resulting in a powerful hybrid author fingerprint.

---

# 🤖 Machine Learning Model

The final classifier is:

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
```

### Why Random Forest?

- Handles High-Dimensional Data
- Robust Against Overfitting
- Works Well with Sparse Features
- Strong Performance on Multi-Class Classification

---

# 📊 Training Pipeline

```text
Dataset
   │
   ▼
Text Cleaning
   │
   ▼
Feature Extraction
   │
   ├── Stylometric Features
   └── TF-IDF Features
   │
   ▼
Feature Fusion
   │
   ▼
Train-Test Split
(80/20)
   │
   ▼
Label Encoding
   │
   ▼
Random Forest Training
   │
   ▼
Model Evaluation
   │
   ▼
Model Serialization
```

---

# 💾 Model Persistence

The trained pipeline saves:

```python
model.pkl
vectorizer.pkl
scaler.pkl
label_encoder.pkl
```

using Joblib.

```python
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
```

---

# 🌐 Backend API

Flask provides REST APIs for inference.

Example:

```http
POST /predict
```

Request:

```json
{
  "text": "Your literary passage..."
}
```

Response:

```json
{
  "author": "Charles Dickens",
  "confidence": 0.93
}
```

---

# 🖥️ Frontend

Built using:

- React.js
- Vite
- Tailwind CSS

Features:

- Modern Cyberpunk UI
- Real-Time Predictions
- Responsive Design
- Interactive Analysis Interface

---

# 🛠️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/scriptmind-ai.git
cd scriptmind-ai
```

---

## Backend Setup

```bash
cd backend

python -m venv venv

source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Flask Server:

```bash
python app.py
```

---

## Frontend Setup

```bash
cd frontend

npm install

npm run dev
```

---

# 📈 Future Improvements

- Deep Learning Based Author Attribution
- Transformer Models (BERT/RoBERTa)
- Confidence Calibration
- Explainable AI Dashboard
- Author Similarity Heatmaps
- Writing Style Visualization
- Multi-Language Support
- Top-K Author Predictions

---

# 🎯 Learning Outcomes

Through this project I gained practical experience in:

- Natural Language Processing
- Stylometry
- Feature Engineering
- Machine Learning Pipelines
- Sparse Matrix Optimization
- Model Serialization
- Flask API Development
- React Frontend Development
- End-to-End ML Deployment

---

# 👨‍💻 Author

**Anand Maurya**

AI Engineer | Machine Learning Enthusiast | NLP Developer

LinkedIn: https://www.linkedin.com/in/anandm2004/

GitHub: https://github.com/Anandmaurya321/AI_Authorization_and_Stylometric_Detection

---

## ⭐ If you found this project interesting, consider giving it a star.
