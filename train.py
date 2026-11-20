
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from config import (
    IMAGE_SIZE,
    MODEL_PATH,
    OUTPUT_DIR,
    EPOCHS,
    LEARNING_RATE
)

from preprocessing import (
    load_training_data,
    load_validation_data
)


def build_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(224, 224, 3)
        ),

        # Image preprocessing
        tf.keras.layers.Rescaling(1./255),

        # Data augmentation
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),

        # First CNN layer
        tf.keras.layers.Conv2D(
            32, (3, 3),
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Second CNN layer
        tf.keras.layers.Conv2D(
            64, (3, 3),
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Third CNN layer
        tf.keras.layers.Conv2D(
            128, (3, 3),
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Fourth CNN layer
        tf.keras.layers.Conv2D(
            256, (3, 3),
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Flatten features
        tf.keras.layers.Flatten(),

        # Fully connected layer
        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.5),

        # Binary classification
        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model():

    print("Loading training images...")

    train_data = load_training_data()
    val_data = load_validation_data()

    model = build_model()

    model.summary()

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3
        )
    ]

    print("Starting model training...")

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save(MODEL_PATH)

    print("Model training completed!")
    print("Saved model:", MODEL_PATH)

    # Plot accuracy
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "training_accuracy.png"
        )
    )

    plt.close()

    # Plot loss
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "training_loss.png"
        )
    )

    plt.close()


if __name__ == "__main__":
    train_model()
