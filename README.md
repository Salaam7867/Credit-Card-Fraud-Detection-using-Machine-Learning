# 💳 Credit Card Fraud Detection System

An end-to-end machine learning project to detect fraudulent credit card transactions with a strong focus on **business cost optimization**, **class imbalance handling**, and **production deployment readiness**.

---

## 🔍 Problem Statement

Credit card fraud detection is a highly imbalanced classification problem where:
- **False Negatives** (missed frauds) cause direct financial loss to banks
- **False Positives** (incorrectly flagged transactions) annoy genuine customers

The goal is **not just high accuracy**, but an optimal **precision–recall tradeoff** that minimizes overall business cost.

---

## 📊 Dataset

- Dataset: Credit Card Transactions (European cardholders)
- Total transactions: ~284,000
- Fraud rate: ~0.17% (highly imbalanced)

> Note: Features `V1`–`V28` are PCA-transformed for privacy.

---

## 🛠️ Approach

### 1. Data Preprocessing
- Train–test split with stratification
- Standard scaling for Logistic Regression
- Careful handling of extreme class imbalance

### 2. Feature Engineering
- Log transformation of transaction amount
- High-amount flag (top percentile transactions)
- Interaction features between amount and PCA components

### 3. Models Trained
- **Logistic Regression** (baseline, interpretable)
- **XGBoost** (benchmark model)

### 4. Threshold Optimization
- Model predictions converted to probabilities
- Decision threshold tuned to prioritize **fraud recall**
- Business-driven selection instead of default 0.5 threshold

### 5. Business Cost Analysis
Assumed costs:
- Fraud missed (FN): ₹10,000
- False alert (FP): ₹500

Threshold selected to minimize **total expected cost**, not just error rate.

---

## 📈 Results (Final)

| Model | Precision | Recall | False Positives |
|------|----------|--------|----------------|
| Logistic Regression | **0.51** | **0.88** | **81** |
| XGBoost | 0.49 | 0.88 | 90 |

**Logistic Regression was chosen for deployment** as it reduced customer friction while maintaining the same fraud capture rate.

---

## 🚀 Deployment (Planned)

- Model artifacts saved (`.pkl`)
- Streamlit app for real-time fraud prediction
- Deployment planned using Streamlit Cloud
- Designed for easy extension to real-time transaction streams

---

## 📁 Project Structure
├── Trained_model.ipynb      # Final Logistic Regression pipeline
├── Comparing XGB & LR/      # Model comparison notebook
├── fraud_model.pkl          # Saved trained model
├── scaler.pkl               # Feature scaler
├── threshold.pkl            # Optimized decision threshold
├── feature_columns.pkl      # Feature schema
├── requirements.txt
└── README.md


---

## 💡 Key Learnings

- Accuracy is misleading for imbalanced problems
- Threshold tuning is critical in real-world ML
- Simpler models can outperform complex ones at the right operating point
- Business cost matters more than leaderboard metrics

---

## 👤 Author

**Mohd Abdul Salaam**  
B.E. Computer Science Engineering  
Aspiring Machine Learning / AI Engineer
