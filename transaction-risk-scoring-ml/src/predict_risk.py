"""Use the saved model to predict the risk of a sample transaction."""

from pathlib import Path
import pickle

import pandas as pd

from preprocessing import prepare_features


def get_risk_level(risk_score):
    """Convert a numeric risk score into a simple text label."""
    if risk_score < 0.33:
        return "Low"
    if risk_score < 0.66:
        return "Medium"
    return "High"


def predict_sample_transaction():
    """Load the model, score a sample transaction, and print the result."""
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "risk_model.pkl"

    with open(model_path, "rb") as model_file:
        saved_model = pickle.load(model_file)

    model = saved_model["model"]
    feature_columns = saved_model["feature_columns"]

    # This is a sample transaction we want to evaluate.
    sample_transaction = pd.DataFrame(
        [
            {
                "amount": 42000,
                "time": 2,
                "location": "international",
                "merchant_type": "electronics",
            }
        ]
    )

    # Apply the same preprocessing steps used during training.
    prepared_transaction = prepare_features(
        sample_transaction,
        expected_columns=feature_columns,
    )

    # predict_proba returns the probability for each class.
    risk_score = model.predict_proba(prepared_transaction)[0][1]
    risk_level = get_risk_level(risk_score)

    print("Transaction details:")
    print(sample_transaction.to_string(index=False))
    print(f"\nRisk score: {risk_score:.2f}")
    print(f"Risk level: {risk_level}")


if __name__ == "__main__":
    predict_sample_transaction()
