import os
import cv2
import numpy as np
import onnxruntime

class ClassificationModel:
    def __init__(self, path):
        self.initialize_model(path)
    
    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path, providers=onnxruntime.get_available_providers())
        # Get model info
        self.get_input_details()
        self.get_output_details()
    
    def classify_roi(self, roi):
        input_tensor = self.prepare_input(roi)
        # Perform inference on the ROI
        outputs = self.inference(input_tensor)
        # Process the classification results
        class_scores = np.squeeze(outputs[0])
        class_id = np.argmax(class_scores)
        class_score = class_scores[class_id]
        return class_id, class_score
    
    def prepare_input(self, roi):
        input_height, input_width = self.input_height, self.input_width
        # Resize ROI to match the input size of the classification model
        roi = cv2.resize(roi, (input_width, input_height))
        # Scale ROI pixel values to 0 to 1
        roi = roi / 255.0
        roi = roi.transpose(2, 0, 1)
        input_tensor = roi[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor
    
    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs
    
    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
    
    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

classification_names = ["Female", "Male"]
model_path = "section.onnx"
Cls = ClassificationModel(model_path)

# Read the input image
image_file = "Specimen_001.tif"
frame = cv2.imread(image_file)

# Classify the image
class_id_cls, class_score = Cls.classify_roi(frame)
classification_class = classification_names[class_id_cls]

# Put the classification result on the image
text = f"Gender: {classification_class} ({class_score:.2f})"
cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display the image with the classification result
cv2.imshow("Classification Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
