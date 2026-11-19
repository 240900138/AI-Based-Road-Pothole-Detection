[README (2).md](https://github.com/user-attachments/files/35100009/README.2.md)
# AI-Based-Road-Pothole-Detection# 🛣️ AI-Based Road Pothole Detection System

A machine learning application that classifies road images as **Pothole** or **Normal Road** using a Convolutional Neural Network (CNN), with a Streamlit web interface for uploading images and viewing results.



---

## Features

- **CNN image classifier** built with TensorFlow / Keras (binary classification)
- **Streamlit web app** with five pages:
  - **Dashboard**: summary metrics and system overview
  - **Detect Potholes**: upload a JPG/PNG road image and get a prediction, confidence score and recommendation
  - **Detection History**: session log of analysed images, downloadable as CSV
  - **Model Performance**: classification report, training curves and confusion matrix
  - **About System**
- **Training pipeline** with data augmentation, early stopping, learning-rate reduction and best-model checkpointing
- **Evaluation script** that produces a classification report and confusion matrix

---

## Project Structure

```
AI Road Pothole Detection/
├── app.py                  # Streamlit web application
├── config.py               # Paths, image size, batch size, epochs, learning rate
├── download_dataset.py     # Downloads the dataset from Kaggle into data/
├── preprocessing.py        # Dataset loading helpers
├── train.py                # Builds and trains the CNN, saves model and plots
├── evaluate.py             # Evaluates the model on the test set
├── predict.py              # Single-image prediction used by the app
├── test.py                 # (currently empty)
├── requirements.txt        # Python dependencies
├── data/
│   ├── train/{Normal,Pothole}/
│   ├── val/{Normal,Pothole}/
│   └── test/               # Flat folder: normal*.jpg and pothole*.jpg
├── models/
│   └── pothole_model.keras # Trained model
└── outputs/
    ├── classification_report.txt
    ├── confusion_matrix.png
    ├── training_accuracy.png
    └── training_loss.png
```

---

## Dataset

Source: [Pothole Detection Dataset on Kaggle](https://www.kaggle.com/datasets/abhinavkulshreshth/pothole-detection-dataset)

| Split      | Normal | Pothole | Total |
|------------|-------:|--------:|------:|
| Train      | 293    | 864     | 1,157 |
| Validation | 46     | 62      | 108   |
| Test       | 71     | 65      | 136   |

The training set is imbalanced (roughly 3 pothole images for every 1 normal image).

In the test folder, labels are taken from the filename prefix (`normal…` or `pothole…`).

---

## Model Architecture

| Stage             | Details                                                    |
|-------------------|------------------------------------------------------------|
| Input             | 224 × 224 RGB image                                        |
| Preprocessing     | Rescaling (1/255)                                          |
| Augmentation      | Random horizontal flip, rotation (0.1), zoom (0.1)         |
| Convolution       | 4 × [Conv2D (32 → 64 → 128 → 256 filters, 3×3, ReLU) + MaxPooling 2×2] |
| Classifier        | Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(1, sigmoid) |
| Loss / Optimiser  | Binary cross-entropy / Adam (learning rate 0.001)          |

**Training settings** (see `config.py`): batch size 32, up to 20 epochs, with `EarlyStopping` (patience 5), `ReduceLROnPlateau` (factor 0.5, patience 3) and `ModelCheckpoint` on validation loss.

A prediction ≥ 0.5 is classified as **Pothole**; otherwise **Normal Road**.

---

## Results

Evaluated on the 136-image test set:

**Overall accuracy: 83.82%**

| Class   | Precision | Recall | F1-score | Support |
|---------|----------:|-------:|---------:|--------:|
| Normal  | 0.95      | 0.73   | 0.83     | 71      |
| Pothole | 0.77      | 0.95   | 0.85     | 65      |

The model is good at catching potholes (95% recall) but tends to over-flag normal roads as potholes (73% recall on Normal). This is a reasonable trade-off for an inspection-assist tool, but it means some false alarms should be expected.

Training curves and the confusion matrix are available in the `outputs/` folder and on the app's **Model Performance** page.

---

## Getting Started

### Prerequisites

- Python 3.9+ (the project was developed with Python 3.13)
- pip

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies: `tensorflow`, `streamlit`, `numpy`, `pandas`, `pillow`, `matplotlib`, `scikit-learn`, `seaborn`, `kagglehub`.

### 2. Prepare the dataset

The dataset is already included in `data/`. To re-download it from Kaggle:

```bash
python download_dataset.py
```

### 3. Train the model

```bash
python train.py
```

Saves the model to `models/pothole_model.keras` and the training plots to `outputs/`.

### 4. Evaluate the model

```bash
python evaluate.py
```

Saves `classification_report.txt` and `confusion_matrix.png` to `outputs/`.

### 5. Run the web app

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (usually http://localhost:8501), go to **Detect Potholes**, upload a road image and click **Analyse Road Image**.

---

## Using the Model from Python

```python
from PIL import Image
from predict import predict_pothole

result = predict_pothole(Image.open("road.jpg"))
print(result)
# {'prediction': 'Pothole Detected', 'confidence': 97.42, 'recommendation': '...'}
```

---

## Technologies

Python · TensorFlow / Keras · Streamlit · NumPy · Pandas · Pillow · Matplotlib · Seaborn · Scikit-learn · KaggleHub

---

## Limitations and Future Improvements

- Small dataset with class imbalance; consider class weighting, more normal-road images, or transfer learning (e.g. MobileNetV2, EfficientNet) to improve accuracy.
- Image-level classification only: it says whether a pothole is present, not where. Object detection (e.g. YOLO) would add localisation and counting.
- Detection history is stored per browser session and is lost when the app is restarted (CSV export is available).
- GPS tagging and integration with municipal reporting systems would make the tool more practical for real-world use.

---

## Acknowledgements

Dataset by [Abhinav Kulshreshth](https://www.kaggle.com/datasets/abhinavkulshreshth/pothole-detection-dataset) on Kaggle.
