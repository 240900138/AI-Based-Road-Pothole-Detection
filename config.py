
import os

# Project directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "pothole_model.keras"
)

# Image settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Training settings
EPOCHS = 20
LEARNING_RATE = 0.001

# Classes
CLASS_NAMES = ["normal", "pothole"]

# Create required directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
