# SmileScan_PredictiveDentalDetection-MobileNetV2

## Overview
This project implements a deep learning model to classify dental problems from images. The model is based on the MobileNetV2 architecture pre-trained on ImageNet and fine-tuned on a custom dataset of dental images. The classification task includes five classes: Tartar, Caries, Gingivitis, Ulcer, and Tooth Discoloration.

## Dataset
The dataset used in this project is sourced from Kaggle, specifically the Oral Diseases Dataset. It consists of images categorized into:

* Calculus
* Gingivitis
* Mouth Ulcer
* Tooth Discoloration
* Dental Caries

Dataset: Oral Diseases  \
Source: Kaggle ([https://www.kaggle.com/datasets/salmansajid05/oral-diseases](https://www.kaggle.com/datasets/salmansajid05/oral-diseases))

## Requirements
* Python 3.x
* TensorFlow 2.x
* NumPy
* OpenCV
* Matplotlib
* Pillow
* tqdm \
Install the required packages using pip install -r requirements.txt.

## Usage
* Data Preparation: Ensure your dataset is structured as per the instructions in the code. Adjust paths and directory structures accordingly. 
* Training: Run the script to train the model using the configured architecture and settings. 
* Evaluation: Evaluate the model's performance on the test set using the evaluation script. 
* Prediction: Use the prediction script to make predictions on new images.

## Model Architecture
The model architecture is based on MobileNetV2 pre-trained on ImageNet, with additional layers for classification:

* Global Average Pooling
* Dense layers with ReLU activation
* Dropout for regularization
* Output layer with Softmax activation for multi-class classification.
* Loss Function: Categorical Crossentropy
* Optimizer: Adam optimizer
* Metrics: Accuracy

## Setup Instructions

1. **Clone the repository**:
```bash
git clone https://github.com/Atharva-Rajan-Kale/SmileScan_PredictiveDentalDetection-MobileNetV2.git
cd SmileScan_PredictiveDentalDetection-MobileNetV2
```
2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download and preprocess datasets**:
Download the dataset and organize it into the data/train/ and data/test/ directories based on the class labels.
Preprocess images, resize them to (224, 224), and save them in NumPy format using np.save.

4. **Training the model**:
Open and run the Jupyter notebook SmileScan_DentalClassification.ipynb.
Train the model using the provided code, adjust hyperparameters as needed, and save the trained model weights.

5. **Testing the model**:
Use the trained model to predict classes on the test dataset.
Evaluate the model performance using metrics like accuracy, confusion matrix, and ROC curves.

## Performance
After training for 30 epochs, the model achieved an accuracy of approximately 94% on the test set.

## Credits
Dataset: Oral Diseases from Kaggle [(link)](https://www.kaggle.com/datasets/salmansajid05/oral-diseases)
