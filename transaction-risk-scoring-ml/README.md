# Transaction Risk Scoring ML

This is a beginner-friendly Python machine learning project that predicts the risk of a financial transaction. The project uses a synthetic dataset and a simple `RandomForestClassifier` model to estimate how likely a transaction is to be fraudulent.

## Project Overview

The goal of this project is to:

- load transaction data from a CSV file
- preprocess the data so a machine learning model can use it
- train a classification model to identify risky transactions
- predict a risk score for a new transaction

The code is written to be easy to read and easy to extend later.

## Project Structure

```text
transaction-risk-scoring-ml/
|- data/
|  `- transactions.csv
|- notebooks/
|  `- risk_analysis.ipynb
|- src/
|  |- preprocessing.py
|  |- train_model.py
|  `- predict_risk.py
|- models/
|- requirements.txt
|- README.md
`- .gitignore
```

## Dataset Description

The dataset is stored in `data/transactions.csv`.

Columns:

- `amount`: transaction amount
- `time`: hour of the day when the transaction happened
- `location`: whether the transaction is local or international
- `merchant_type`: type of merchant involved in the transaction
- `fraud`: target label where `0` means normal and `1` means fraudulent

This is a synthetic educational dataset with 1000 rows created for learning purposes.

## Install Dependencies

1. Open a terminal in the `transaction-risk-scoring-ml` folder.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Train the Model

Run the training script from inside the `transaction-risk-scoring-ml` folder:

```bash
python src/train_model.py
```

This script will:

- load the dataset
- preprocess the transaction data
- split the data into training and testing sets
- train a `RandomForestClassifier`
- print the accuracy score
- save the trained model to `models/risk_model.pkl`

### Example Training Output

When the project was last run on the current 1000-row dataset, the script produced:

```text
Accuracy: 0.76
Model saved to: D:\GitHub\Transaction-Risk-Scoring-ML\transaction-risk-scoring-ml\models\risk_model.pkl
```

This accuracy is reasonable for a beginner demo project using a simple model and synthetic data.

## Predict Transaction Risk

After training the model, run:

```bash
python src/predict_risk.py
```

This script will:

- load the saved model
- create a sample transaction
- calculate the fraud probability using `predict_proba()`
- print the transaction details
- print the risk score from `0` to `1`
- print a simple risk level: `Low`, `Medium`, or `High`

### Example Prediction Output

```text
Transaction details:
 amount  time      location merchant_type
  42000     2 international   electronics

Risk score: 0.99
Risk level: High
```

## Explore the Data

The notebook `notebooks/risk_analysis.ipynb` contains simple exploratory data analysis. It shows how to:

- load the dataset
- view summary statistics
- plot the transaction amount distribution
- compare fraud and normal transactions
