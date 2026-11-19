 AI-Based Road Pothole Detection System

## 1. Introduction

The AI-Based Road Pothole Detection System is a
Machine Learning application designed to identify
potholes using road images.

The system uses Convolutional Neural Networks (CNN)
to classify road images into two categories:

- Pothole
- Normal Road

## 2. Technologies

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## 3. Machine Learning Approach

The system uses supervised learning.

A CNN model is trained using labelled road images.

The model learns visual patterns from the training
dataset and uses those patterns to classify new images.

## 4. Dataset

Source:

https://www.kaggle.com/datasets/abhinavkulshreshth/pothole-detection-dataset

The dataset contains images divided into:

- Training data
- Validation data
- Testing data

## 5. Installation

Install Python and the required libraries.

Run:

pip install -r requirements.txt

## 6. Dataset Preparation

Run:

python download_dataset.py

## 7. Model Training

Run:

python train.py

## 8. Model Evaluation

Run:

python evaluate.py

## 9. Start the Application

Run:

streamlit run app.py

## 10. Expected Output

The system displays:

- Uploaded road image
- Pothole or Normal Road prediction
- Prediction confidence
- Recommendation
- Detection history
- Model evaluation results


