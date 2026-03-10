"""Train a beginner-friendly transaction risk model."""

from pathlib import Path
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from preprocessing import (
    get_default_data_path,
    load_dataset,
    prepare_features,
    separate_features_and_labels,
)


def train_transaction_risk_model():
    """Train the model, print accuracy, and save the result to disk."""
    project_root = Path(__file__).resolve().parents[1]
    data_path = get_default_data_path()
    model_path = project_root / "models" / "risk_model.pkl"

    # Load the CSV file into a pandas DataFrame.
    dataset = load_dataset(data_path)

    # Convert the text columns into numeric columns the model can understand.
    prepared_features = prepare_features(dataset)

    # Separate the input columns from the fraud label column.
    features, labels = separate_features_and_labels(
        prepared_features.assign(fraud=dataset["fraud"])
    )

    # Split the dataset so we can test the model on unseen examples.
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.3,
        random_state=42,
        stratify=labels,
    )

    # RandomForestClassifier is simple to use and works well for this small example.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Measure how often the model predicts the correct class.
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")

    # Save both the model and the feature names for future predictions.
    saved_model = {
        "model": model,
        "feature_columns": list(features.columns),
    }

    with open(model_path, "wb") as model_file:
        pickle.dump(saved_model, model_file)

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train_transaction_risk_model()
