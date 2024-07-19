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
import pytesseract

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
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# Define a function to preprocess Javanese script using OCR
def preprocess_javanese_script(image):
    image = np.array(image)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform OCR to get word bounding boxes
    detection_boxes = pytesseract.image_to_boxes(img_rgb)
    
    # Extract bounding boxes for characters
    segmented_chars = []
    for box in detection_boxes.splitlines():
        b = box.split()
        x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        segmented_chars.append(image[y:h, x:w])
    
    return segmented_chars

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
        char_image_pil = Image.fromarray(char_image)
        char_class = predict(char_image_pil, model, transform)
        recognized_text += char_class + " "
    
    # Display the recognized text
    st.write(f"Recognized Text: {recognized_text}")


