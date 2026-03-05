# 🔐 Network Intrusion Detection System

A Machine Learning-based Network Intrusion Detection System built using Scikit-Learn and Streamlit.

## 🚀 Features

- Binary classification: Normal vs Attack traffic
- Random Forest Classifier
- Scikit-Learn Pipeline with ColumnTransformer
- OneHotEncoder with unseen-label handling
- Interactive Streamlit Web Interface
- Confusion Matrix visualization
- Feature Importance graph
- CSV upload and prediction download

## 🧠 Model Details

- Dataset: NSL-KDD
- Algorithm: Random Forest
- Accuracy: ~76%
- Preprocessing:
  - StandardScaler (numerical features)
  - OneHotEncoder (categorical features)
  - Pipeline-based architecture


Sample Output

Confusion Matrix

Feature Importance visualization

Attack / Normal prediction summary

📌 Tech Stack

Python

Scikit-Learn

Streamlit

Pandas

Matplotlib

## 🖥️ How to Run

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py