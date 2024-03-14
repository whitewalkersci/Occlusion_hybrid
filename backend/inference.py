import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2 
import time 
import os

class CNN:
    def __init__(self, model_path):
        # Load the trained model
        self.model = mobilenet_v3_small()
        num_classes = 3  # Update with the number of classes in your dataset
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Define the transformation for the input image
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Define class list
        self.class_list = ['cell', 'final_Unidentified', 'no_cell']

    # Define the inference function
    def predict_image(self, image):
        # Convert the image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert the image to PIL format
        image_pil = Image.fromarray(image)
        
        image_pil = self.convert_to_rgb(image_pil)
        image_pil = self.test_transform(image_pil).unsqueeze(0)  # Add a batch dimension
        with torch.no_grad():
            output = self.model(image_pil)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()
        predicted_class = self.class_list[predicted_index]  # Get class name from class list
        return predicted_class, probabilities[predicted_index].item()

    # Convert grayscale image to RGB
    def convert_to_rgb(self, image):
        if image.mode == "L":
            image = image.convert("RGB")
        return image


if __name__ == "__main__":
    # Path to the model file
    model_path = "best_mobilenetv3_custom.pth"  # Update with your model path

    # Create an instance of MobileNetV3Inference
    mobilenet_inference = CNN(model_path)

    # Define the image path
    image_path = "example.jpg"  # Update with your image path

    # Read the image using OpenCV
    image_cv2 = cv2.imread(image_path)

    # Perform inference
    predicted_class, confidence = mobilenet_inference.predict_image(image_cv2)
    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")