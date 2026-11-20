
import tensorflow as tf
import numpy as np

from PIL import Image
from config import MODEL_PATH, IMAGE_SIZE


def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


def predict_pothole(image):

    model = load_model()

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image
    image = image.resize(IMAGE_SIZE)

    # Convert image to array
    image_array = np.array(image)

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Make prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]

    # Classification
    if prediction >= 0.5:

        result = "Pothole Detected"

        confidence = prediction * 100

        recommendation = (
            "Road damage detected. "
            "The road should be inspected "
            "by the relevant municipal department."
        )

    else:

        result = "Normal Road"

        confidence = (1 - prediction) * 100

        recommendation = (
            "No pothole was detected in the "
            "uploaded image."
        )

    return {
        "prediction": result,
        "confidence": round(float(confidence), 2),
        "recommendation": recommendation
    }
