# Face Mask Detection using CNN

This project implements a Convolutional Neural Network (CNN) to detect whether individuals are wearing face masks using image data. It classifies images into two categories: **With Mask** and **Without Mask**, helping enforce health compliance policies, especially during outbreaks like COVID-19.


## Contents

- Dataset loading from Kaggle
- Data preprocessing and labeling
- Image augmentation using `ImageDataGenerator`
- CNN model building using Keras/TensorFlow
- Model training and validation
- Evaluation metrics and visualizations


## Classes

| Label | Description     |
|-------|-----------------|
| 0     | With Mask       |
| 1     | Without Mask    |


## 🏗Architecture

- **Model Type**: Custom built CNN
- **Layers**:
  - Convolutional Layers (`Conv2D`) with ReLU activation
  - MaxPooling layers
  - BatchNormalization
  - Dropout layers to reduce overfitting
  - Flatten + Fully Connected Dense layers
- **Output Layer**: `Dense(1, activation='sigmoid')` for binary classification
- **Compilation**:
  - Loss: `binary_crossentropy`
  - Optimizer: `Adam`
  - Metrics: `accuracy`


## Dataset

- **Source**: [Face Mask Dataset - Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- **Structure**: Images are split into two folders:
  - `/with_mask/`
  - `/without_mask/`
- **Preprocessing**:
  - Images resized to 224x224 pixels
  - Normalized to pixel values in range [0, 1]
  - Converted to NumPy arrays
  - Augmented with:
    - Rotation
    - Zoom
    - Horizontal flips
    - Brightness variation


### Train-Test Split

- Training Set: 70%
- Test Set: 30%


## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix


## Results

Example results from the notebook:

- Test Accuracy: 94.79%
- High performance on both training and validation sets
- Confusion matrix shows good class separation
- Precision: 0.95
- Recall: 0.95
- F1 Score: 0.95


## Requirements

Install dependencies with:

```bash
pip install tensorflow keras numpy pandas matplotlib opencv-python scikit-learn seaborn Pillow
