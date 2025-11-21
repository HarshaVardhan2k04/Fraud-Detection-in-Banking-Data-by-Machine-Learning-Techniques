# Fraud Detection in Banking Data by Machine Learning Techniques

This project aims to detect fraudulent transactions in banking data using supervised machine learning algorithms. It addresses the challenge of class imbalance, feature engineering, and model evaluation in fraud detection systems.

---

## 📘 Motivation  
Financial fraud is a major concern in banking and fintech. With digital transactions skyrocketing, it’s critical to detect and prevent fraudulent activity in real time. This project builds an end-to-end pipeline to analyze transaction patterns and classify them as **fraudulent** or **legitimate**.

---

## 🔍 Project Overview  
The pipeline includes:  
- Data loading & cleaning  
- Feature engineering for transaction behavior  
- Handling class imbalance (undersampling/oversampling)  
- Training ML models (e.g., Random Forest, XGBoost, SVM)  
- Evaluating using precision, recall, F1-score, ROC-AUC  
- Saving models and reporting results  

---

## 📂 Suggested Folder Structure  
Fraud-Detection-in-Banking-Data-by-Machine-Learning-Techniques/
│
├── data/
│ ├── raw/ # Original datasets
│ └── processed/ # Cleaned & feature-engineered data
│
├── notebooks/ # Jupyter notebooks for exploration & modeling
│ └── 01_exploration.ipynb
│
├── src/
│ ├── preprocessing.py # Data cleaning functions
│ ├── feature_engineering.py # Feature creation
│ ├── train_models.py # Model training scripts
│ └── evaluate.py # Evaluation and reporting
│
├── models/ # Saved model files (.pkl, .joblib)
│
├── results/ # Reports, confusion matrices, metrics
│
├── requirements.txt
├── README.md
└── LICENSE

yaml
Copy code

---

## 🧪 Dataset  
The project uses publicly available banking/credit‐card transaction data (e.g., **Credit Card Fraud Detection**).  
Key challenge: Fraud cases are rare → creates large class imbalance that must be handled carefully.

---

## ✏️ Feature Engineering  
Typical features include:  
- Transaction amount  
- Time since last transaction  
- Aggregated customer behavior  
- Derived ratios and statistical summaries  
- Encoding categorical transaction attributes  

---

## 🤖 Machine Learning Models Used  
We experiment with:  
- Logistic Regression  
- Random Forest  
- XGBoost  
- Support Vector Machine (SVM)  
- Gradient Boosting  

We optimize hyperparameters and compare models based on metrics like recall (critical for detecting fraud) and precision (to reduce false‐positives).

---

## 📊 Evaluation Metrics  
Because fraud detection emphasizes rare events, we consider:  
- Precision  
- Recall  
- F1‐score  
- ROC AUC  
- Confusion matrix  

Always check performance on the minority class (fraudulent transactions).

---

## 🛠️ Installation  
```bash
git clone https://github.com/HarshaVardhan2k04/Fraud-Detection-in-Banking-Data-by-Machine-Learning-Techniques.git
cd Fraud-Detection-in-Banking-Data-by-Machine-Learning-Techniques
python -m venv env
source env/bin/activate        # On Windows: env\Scripts\activate
pip install -r requirements.txt
```
## 👥 Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/HarshaVardhan2k04">
        <img src="https://github.com/HarshaVardhan2k04.png" width="100px;" alt="Harsha"/>
        <br /><sub><b>Harsha Vardhan</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/MOhanNaidu04">
        <img src="https://github.com/MOhanNaidu04.png" width="100px;" alt="Mohan"/>
        <br /><sub><b>Mohan Naidu</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/sreevamsee">
        <img src="https://github.com/sreevamsee.png" width="100px;" alt="Srivamshi"/>
        <br /><sub><b>Srivamshi Voggu</b></sub>
      </a>
    </td>
  </tr>
</table>
