# SQL Injection Detection using Machine Learning 🔐💻

A machine learning-based approach to detect **SQL Injection (SQLi) attacks** using **TF-IDF vectorization**, **Autoencoders** for feature compression, and **XGBoost** for classification.

---

## 🧠 Project Overview

SQL Injection is one of the most dangerous web vulnerabilities, allowing attackers to access or modify data in the database by injecting malicious SQL code into user inputs.

This project leverages Machine Learning to identify such malicious queries and classify them as **safe** or **malicious**.

---

## 🚀 Objectives

- Detect SQL Injection patterns in queries using ML.
- Convert SQL queries to meaningful numerical features using TF-IDF.
- Compress feature space using an Autoencoder neural network.
- Build a robust classifier using XGBoost.
- Evaluate and visualize performance using metrics and plots.

---

## 📁 Dataset

The dataset consists of SQL queries labeled as either:
- `0` → Safe Query
- `1` → SQL Injection (Malicious)

> Preprocessing steps were applied to ensure the data is clean, consistent, and meaningful for machine learning.

---

## 🛠️ Technologies & Libraries Used

- **Python**
- **Pandas**, **NumPy** – Data preprocessing
- **Scikit-learn** – TF-IDF, train-test split, metrics
- **Keras** – Autoencoder (feature compression)
- **XGBoost** – Classifier model
- **Matplotlib**, **Seaborn** – Data visualization

---

## 🧪 Model Architecture

### 🔹 Feature Extraction:
- TF-IDF Vectorizer converts query text into a 544-dimensional vector.
- Tokenization already handled; focus is on capturing keyword importance.

### 🔹 Autoencoder:
A neural network to reduce input features from 544 to a compressed representation of 64.

```python
Input -> Dense(128, relu) -> Dense(64, relu)  ← Encoder  
       -> Dense(128, relu) -> Dense(544, linear) ← Decoder
