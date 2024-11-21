import tensorflow as tf
import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
from io import BytesIO
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Load pre-trained MobileNetV2 model without top layers (we're not using the ImageNet classifier)
base_model = MobileNetV2(weights='imagenet', include_top=False)

# Add custom layers for gender prediction (binary classification)
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global average pooling to reduce dimensionality
x = Dense(1024, activation='relu')(x)  # Custom dense layer
predictions = Dense(1, activation='sigmoid')(x)  # 1 unit for binary classification (male/female)

# Final model
gender_model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all layers except the last few
for layer in base_model.layers[:-10]:  # Unfreeze the last 10 layers
    layer.trainable = False

# Compile the model (binary classification)
gender_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Function to fetch image from URL
def fetch_image_from_url(url):
    """
    Fetches an image from the URL and converts it into a format that OpenCV can handle.
    """
    response = requests.get(url)
    image_data = BytesIO(response.content)  # Convert to BytesIO object
    img = cv2.imdecode(np.asarray(bytearray(image_data.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
    return img

# Step to preprocess and predict gender (using the loaded model)
def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

    # Detect faces in the image
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face = img[y:y + h, x:x + w]

        # Preprocess the face for prediction (resize to input size)
        face = cv2.resize(face, (224, 224))
        face = np.expand_dims(face, axis=0) / 255.0  # Normalize to [0, 1]
        
        # Predict the gender using the pre-trained model
        prediction = gender_model.predict(face)
        gender = 1 if prediction[0] > 0.5 else 0  # 1 for female, 0 for male

        # Draw gender text on the image
        label = "Female" if gender == 1 else "Male"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"Predicted Gender: {label}", (x, y-10), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the processed image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axes
    plt.show()

# Main function to run the program
def main():
    # Step 1: Get image URL from the user
    url = input("Enter the image URL: ")  # Prompt the user to enter an image URL
    
    # Step 2: Fetch the image from the URL
    img = fetch_image_from_url(url)
    
    # Step 3: Process the image to detect faces and predict gender
    img_with_gender = process_image(img)
    
    # Step 4: Display the results
    # display_results(img_with_gender)

if __name__ == "__main__":
    main()
