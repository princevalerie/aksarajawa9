import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageOps
import io
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the trained model
class_names = ['ba', 'ca', 'da', 'dha', 'ga', 'ha', 'ja', 'ka', 'la', 'ma', 
               'na', 'nga', 'nya', 'pa', 'ra', 'sa', 'ta', 'tha', 'wa', 'ya']

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(in_features=512, out_features=20, bias=True)
model.load_state_dict(torch.load('cnn_model1.pth', map_location=torch.device('cpu')))
model.eval()

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Define a function to predict the class
def predict(image, model, transform):
    # Convert the image to RGB
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# Function to preprocess the Javanese script image
def preprocess_javanese_script(image):
    # Convert the image to grayscale
    image = ImageOps.grayscale(image)
    image_np = np.array(image)

    # Thresholding to create a binary image
    _, binary_image = cv2.threshold(image_np, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # Finding contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Bounding box extraction
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Sorting the bounding boxes from left to right, top to bottom
    bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1], b[0]))

    characters = []
    for bbox in bounding_boxes:
        x, y, w, h = bbox
        char_image = binary_image[y:y+h, x:x+w]
        char_pil_image = Image.fromarray(char_image)
        characters.append(char_pil_image)

    return characters

# Function to detect spaces between words
def detect_spaces(bounding_boxes, threshold=20):
    words = []
    current_word = []
    for i, bbox in enumerate(bounding_boxes):
        if i == 0:
            current_word.append(bbox)
            continue
        previous_bbox = bounding_boxes[i - 1]
        x_prev, y_prev, w_prev, h_prev = previous_bbox
        x, y, w, h = bbox
        # Calculate horizontal gap between the current and previous bounding box
        horizontal_gap = x - (x_prev + w_prev)
        if horizontal_gap > threshold:
            words.append(current_word)
            current_word = []
        current_word.append(bbox)
    if current_word:
        words.append(current_word)
    return words

# Function to plot words
def plot_words(image, words):
    plt.figure(figsize=(10, 5))
    for i, word in enumerate(words):
        # Create a blank canvas for each word
        word_canvas = np.zeros_like(image)
        for bbox in word:
            x, y, w, h = bbox
            word_canvas[y:y+h, x:x+w] = image[y:y+h, x:x+w]
        plt.subplot(1, len(words), i + 1)
        plt.imshow(word_canvas, cmap='gray')
        plt.axis('off')
    plt.show()

# Streamlit app
st.title("Aksara Jawa Detection")

# Camera input
image_data = st.camera_input("Take a picture")

if image_data is not None:
    # Load the image
    image = Image.open(io.BytesIO(image_data.getvalue()))
    
    # Display the image
    st.image(image, caption='Captured Image', use_column_width=True)
    
    # Segment characters from the image
    segmented_chars = preprocess_javanese_script(image)
    
    # Predict each character
    recognized_text = ""
    for char_image in segmented_chars:
        char_class = predict(char_image, model, transform)
        recognized_text += char_class + " "
    
    # Display the recognized text
    st.write(f"Recognized Text: {recognized_text}")
