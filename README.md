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
# Re-creating the README.md file since the code execution environment has been reset.


### 🔹 Classifier:

- XGBoost model is trained on encoded features to predict malicious (1) or safe (0).

---

## 🔍 Model Evaluation

- **Accuracy**: ~99.5% on validation data
- **Confusion Matrix**
- **Classification Report**
- **Loss Graph** (for autoencoder)

### Example Output:
```
Train Accuracy: 0.9978
Validation Accuracy: 0.9957
Confusion Matrix:
 [[4082    3]
  [  35 4666]]
```

---

## 📊 Visualizations

- Training vs Validation Loss for Autoencoder
- Confusion Matrix heatmap
- ROC Curve (optional extension)

---

## 🔐 Why ML for SQLi?

> Traditional rule-based methods (like `if-else` checks) are limited.

Machine Learning:

- **Learns patterns** from real malicious and safe queries
- **Generalizes** to detect new, unseen SQLi attacks
- Works well even if attacker slightly **modifies query**

---

## 💬 Sample Prediction

```python
query = "SELECT * FROM users WHERE username='admin' --'"
# Output: Prediction: 1 (SQL Injection)
```

---

## 🧠 Key Learnings

- How autoencoders can compress high-dimensional textual features.
- Use of TF-IDF in cybersecurity context.
- How to combine unsupervised and supervised learning for anomaly detection.
- Real-world implications of data preprocessing and feature engineering.

---

## 📂 Folder Structure

```
SQLi-Detection-ML/
├── data/
│   └── sql_queries.csv
├── models/
│   ├── autoencoder_model.h5
│   └── xgboost_model.pkl
├── images/
│   └── loss_graph.png
├── main_notebook.ipynb
└── README.md
```

---

## 🧪 How to Run

1. Clone this repository:

```bash
git clone https://github.com/your-username/sqli-detection-ml.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook:

```bash
jupyter notebook main_notebook.ipynb
```

---

## 📌 Future Work

- Use deep learning models like LSTM or BERT for sequence learning.
- Create a Streamlit or Flask web app for real-time prediction.
- Add support for detecting other types of web attacks (XSS, CSRF).

---

## 🙌 Acknowledgements

Thanks to open datasets and resources that helped in making this project.  
Special thanks to the community for tools like Scikit-learn, Keras, and XGBoost.

---

## 🧑‍💻 Author

**Koushik Samudrala**  
🎓 B.Tech CSE @ SASTRA University  
💡 Passionate about Web Security, AI, and Software Development  
🌍 Goal: Travel the world 🌏  
📫 LinkedIn | GitHub

---

## 📃 License

This project is licensed under the MIT License.

---
