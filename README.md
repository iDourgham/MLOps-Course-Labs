# Bank Consumer Churn Prediction

This project focuses on predicting customer churn for a bank using machine learning models. The objective is to identify which customers are likely to leave the bank, based on their historical data and attributes.

## 📌 Project Overview

The dataset used is [`Churn_Modelling.csv`](data/Churn_Modelling.csv), which contains demographic and transactional data of bank customers, along with a binary target variable `Exited` indicating whether a customer has left the bank.

### Key Features Used
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Ownership
- Active Membership Status
- Estimated Salary

---

## 🧠 Models Tried

Three models were trained and evaluated:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Support Vector Classifier (SVC)**

### ✅ Final Model Choice: SVC

After evaluating all models using key metrics — **Accuracy, Precision, Recall, and F1-score** — the **Support Vector Classifier (SVC)** showed the best overall performance. Thus, we chose to proceed with the SVC model for deployment.

---

## ⚙️ Project Structure

├── data/\
│ └── Churn_Modelling.csv\
├── model/\
│ └── model.pkl\
├── src/\
│ └── train.py\
├── column_transformer.pkl\
├── confusion_matrix.png\
├── README.md

---

## 📦 Dependencies

This project uses Python and the following libraries:

- `pandas`

- `scikit-learn`

- `matplotlib`

- `mlflow`

- `joblib`

To install dependencies:

```bash

pip install -r requirements.txt

---

If you're using Conda, activate the environment:

bash
`conda activate ./.churn_prediction`
