
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from config import MODEL_PATH, OUTPUT_DIR
from preprocessing import load_flat_test_data


def evaluate_model():

    print("Loading trained model...")

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "Trained model not found. Run train.py first."
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Loading test images...")

    images, labels = load_flat_test_data()

    print("Total test images:", len(labels))

    print("Predicting potholes...")

    predictions = model.predict(
        images,
        batch_size=32,
        verbose=1
    )

    predicted = (
        predictions.flatten() >= 0.5
    ).astype(int)

    actual = labels.numpy()

    # Accuracy
    accuracy = accuracy_score(
        actual,
        predicted
    )

    print("\nMODEL ACCURACY")
    print(f"{accuracy * 100:.2f}%")

    # Classification report
    report = classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["Normal", "Pothole"],
        zero_division=0
    )

    print("\nCLASSIFICATION REPORT")
    print(report)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    with open(
        os.path.join(
            OUTPUT_DIR,
            "classification_report.txt"
        ),
        "w"
    ) as file:

        file.write(
            f"Model Accuracy: {accuracy * 100:.2f}%\n\n"
        )

        file.write(report)

    # Confusion matrix
    cm = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Pothole"],
        yticklabels=["Normal", "Pothole"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.title(
        "AI-Based Road Pothole Detection"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "confusion_matrix.png"
        )
    )

    plt.close()

    print("\nConfusion matrix saved.")

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":

    evaluate_model()
