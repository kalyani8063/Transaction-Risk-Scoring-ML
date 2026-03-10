"""Helper functions for loading and preparing transaction data."""

from pathlib import Path

import pandas as pd


def load_dataset(csv_path):
    """Load the transaction dataset from a CSV file."""
    dataset = pd.read_csv(csv_path)
    return dataset


def encode_categorical_variables(dataset):
    """Convert text columns into numeric columns using one-hot encoding."""
    encoded_dataset = pd.get_dummies(
        dataset,
        columns=["location", "merchant_type"],
        drop_first=False,
    )
    return encoded_dataset


def separate_features_and_labels(dataset, label_column="fraud"):
    """Split the dataset into input features and target labels."""
    features = dataset.drop(columns=[label_column])
    labels = dataset[label_column]
    return features, labels


def prepare_features(dataset, expected_columns=None):
    """
    Prepare model input features.

    If expected_columns are provided, the function adds any missing columns
    and reorders the final DataFrame so it matches the training data.
    """
    encoded_dataset = encode_categorical_variables(dataset)

    if "fraud" in encoded_dataset.columns:
        encoded_dataset = encoded_dataset.drop(columns=["fraud"])

    if expected_columns is not None:
        encoded_dataset = encoded_dataset.reindex(columns=expected_columns, fill_value=0)

    return encoded_dataset


def get_default_data_path():
    """Return the default path to the sample dataset."""
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "data" / "transactions.csv"
