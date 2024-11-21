# Gender Prediction using MobileNetV2 and OpenCV

This project uses the pre-trained MobileNetV2 model combined with OpenCV to predict the gender (male or female) of a person based on their face in the image. The model leverages MobileNetV2 for feature extraction and OpenCV’s Haar Cascade Classifier for face detection. The project fetches an image from a URL, detects faces, preprocesses the image, and classifies the gender (male or female) using a custom classifier.


[Overview](#overview)

[Techniques Covered](#techniques-covered)

[Features](#features)

[Usage](#usage)

[Dependencies](#dependencies)

[Results](#results)


## OverView
The goal of this project is to predict gender from images of faces using MobileNetV2, a lightweight deep learning model, efficient for mobile and embedded vision applications, along with OpenCV for detecting faces. The MobileNetV2 model is pre-trained on the ImageNet dataset and fine-tuned for binary gender classification (male/female). The project fetches an image from a URL, detects faces using OpenCV, preprocesses the image, and then classifies the gender based on the detected face.

## Techniques Covered
- MobileNetV2: Pre-trained deep learning model used for feature extraction and classification.
- OpenCV Face Detection: Using Haar Cascade Classifier from OpenCV to detect faces in the image.
- Image Preprocessing: Resizing and normalizing the image to fit the model’s input requirements.
- Binary Classification: Predicting male or female using a sigmoid activation function for gender classification.

## Features
- Image URL Input: Allows users to input an image URL for gender prediction.
- Pre-trained MobileNetV2: Utilizes MobileNetV2 pre-trained on ImageNet for feature extraction.
- Face Detection with OpenCV: Detects faces in the image using OpenCV’s Haar Cascade Classifier.
- Matplotlib Visualization: Displays the image with the predicted gender label.

## Usage
- Fetch Image from URL: Input an image URL to fetch the image from the web.
- Preprocess Image: Resize and normalize the image to fit the MobileNetV2 model’s input size.
- Detect Face and Classify Gender: Detect faces in the image using OpenCV and predict the gender using MobileNetV2.
- Display Results: The image is displayed with the predicted gender label (male or female).

## Step-by-step
- Enter the Image URL: Provide the image URL for the model to classify.
- Preprocess the Image: The image is resized to 224x224 pixels and normalized.
- Face Detection: Detect faces in the image using OpenCV’s Haar Cascade Classifier.
- Model Classification: Use MobileNetV2 to predict the gender (male or female).
- Display Results: The image with the predicted gender (male or female) is displayed.
  
## Dependencies
```
requests         # Fetching image from the URL
tensorflow       # TensorFlow for loading and using the MobileNetV2 model
opencv-python    # For face detection using Haar Cascade Classifier
matplotlib       # Displaying the image and results
numpy            # Numerical computations
Pillow           # Image processing

```
## Results
- Prediction Accuracy: The model predicts the gender (male or female) based on the face in the image.
- Gender Classification: The model predicts whether the person in the image is male or female.

### Sample Output

#### Image Display
The uploaded image is displayed with the predicted gender label (Male or Female).

#### Gender Prediction
```
Predicted Gender: Male
```

