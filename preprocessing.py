
import tensorflow as tf
import os

from config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE
)


def get_class_names(directory):

    folders = os.listdir(directory)

    folder_map = {
        folder.lower(): folder
        for folder in folders
        if os.path.isdir(
            os.path.join(directory, folder)
        )
    }

    normal = folder_map.get("normal")
    pothole = folder_map.get("pothole")

    if normal is None or pothole is None:

        raise ValueError(
            f"Class folders not found in {directory}"
        )

    return [normal, pothole]


def load_dataset(directory, shuffle):

    classes = get_class_names(directory)

    dataset = tf.keras.utils.image_dataset_from_directory(

        directory,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        label_mode="binary",

        class_names=classes,

        shuffle=shuffle,

        seed=42
    )

    return dataset


def load_training_data():

    return load_dataset(TRAIN_DIR, True)


def load_validation_data():

    return load_dataset(VAL_DIR, False)


def load_test_data():

    return load_dataset(TEST_DIR, False)


# Load test images stored directly in one folder
def load_flat_test_data():

    images = []
    labels = []

    for filename in sorted(os.listdir(TEST_DIR)):

        filepath = os.path.join(TEST_DIR, filename)

        if not os.path.isfile(filepath):
            continue

        name = filename.lower()

        if name.startswith("normal"):
            label = 0

        elif name.startswith("pothole"):
            label = 1

        else:
            continue

        try:

            image = tf.keras.utils.load_img(
                filepath,
                target_size=IMAGE_SIZE
            )

            image = tf.keras.utils.img_to_array(
                image
            )

            images.append(image)
            labels.append(label)

        except Exception as error:

            print("Could not load:", filename, error)

    if len(images) == 0:

        raise ValueError(
            "No labelled test images were found."
        )

    return (
        tf.convert_to_tensor(images, dtype=tf.float32),
        tf.convert_to_tensor(labels, dtype=tf.int32)
    )
