# MicroOrganism Image Classification with Azure ML

## Overview

This project implements a MicroOrganism Image Classification system using a CNN. The Azure ML pipeline notebook sets up an efficient inference process.

## Methodology

The project workflow encompasses the following phases:

1. **Data Collection and Preprocessing:**
   - Retrieved the dataset from Kaggle, consisting of images categorized under the eight distinct microorganism classes.
   - Performed data cleaning, resizing, and normalization to prepare the images for model training.

2. **Model Development:**
   - Utilized a convolutional neural network (CNN) architecture or custom models to train the classification model.

3. **Model Training and Validation:**
   - Split the dataset into training and validation sets.
   - Trained the model on the training data, validating its performance on the validation set.
   - Fine-tuned hyperparameters for optimal accuracy and generalization.

4. **Model Evaluation and Testing:**
   - Assess the model's performance using various evaluation metrics.
   - Test the model with unseen data to ensure its ability to generalize and accurately classify microorganisms.

5. **Azure ML Pipeline Integration (Inference):**
   - Created an Azure ML pipeline specifically designed for model inference.
   - Automated and managed the steps involved in the inference process.
   - Organized and executed the model's inference seamlessly within this dedicated pipeline.


## Structure

- **MicroOrganism_Image_Classification_CNN.ipynb**: Jupyter notebook for image classification model implementation.
- **MicroOrganism_Classification_Inference_Pipeline_AzureML.ipynb**: Notebook for Azure ML inference pipeline setup.
- **Dataset**: [Microorganism Classification Dataset](https://www.kaggle.com/datasets/mdwaquarazam/microorganism-image-classification)


## Prerequisites

- Python (v3.7.3 or higher)
- Libraries: Keras, Scikit-learn, TensorFlow, seaborn, matplotlib, Azure ML SDK.

## Getting Started

1. Clone: `git clone https://github.com/AnkitaMungalpara/MicroOrganism-Image-Classification-AzureML.git`
2. Run notebooks.

## Usage

- Train the image classification model using the CNN notebook.
- Set up a scalable inference process with the Azure ML notebook.
