import cv2
import os
import numpy as np
from tqdm import tqdm
from backend.model_inference import ClassificationModel
import time 
from backend.inference import CNN 
class OcclusionModel:
    def __init__(self, model_path,model_filter_path):
        self.model = ClassificationModel(model_path,['cell','no_cell','UNKNOWN'])
        self.model_filter = CNN(model_filter_path)


    def check_and_convert_to_bgr(self,image):
        # Check if the image is grayscale
        if len(image.shape) == 2:
            # Convert grayscale image to BGR
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            return image


    def occlusion_finder(self, pillar_points, section_image, section_name,sp_name):
        positive = 0
        negative = 0

        if section_name == 0:
            offset = 6
        elif section_name == 1:
            offset = 3
        elif section_name == 2:
            offset = -4
        else:
            offset = -6


        for id,p in enumerate(pillar_points):

            x, y, w, h = p[0], p[1], p[2], p[3]
            # input_crop = section_image[y1:y2, x1:x2]

            x1,y1 = x, y
            x2,y2 = x+w,y+h

            
            x1= max(x1-w//2,0)
            y1 = y1 + 3
            y2 = int(y2+3*w//4) + offset

            input_crop = section_image[y1:y2, x1:x2]
            
            scaling_factor = 255.0 / np.max(input_crop)
            input_crop = (input_crop * scaling_factor).astype(np.uint8)


            input_crop = self.check_and_convert_to_bgr(input_crop)

            classification_class = self.model.classify_roi(input_crop)
            # progress_bar.update(1)
            if classification_class == 'cell':
                
                predicted_class, confidence = self.model_filter.predict_image(input_crop)
                if predicted_class == 'cell':
                    positive += 1
                    # os.makedirs(f'crop/{sp_name}/{section_name}/cell',exist_ok=True)
                    # cv2.imwrite(f'crop/{sp_name}/{section_name}/cell/{id}.jpg',input_crop)
                    directory_true = os.path.join(os.getcwd(),'database', sp_name,'occlusion_images','section-'+str(4-section_name), 'cell')
                    os.makedirs(directory_true, exist_ok=True)
                    
                    file_path = os.path.join(directory_true, f'{sp_name}-{section_name}-{confidence}-{positive}.jpg')
                    cv2.imwrite(file_path, input_crop)
                else:
                    directory_false = os.path.join(os.getcwd(),'database', sp_name,'occlusion_images','section-'+str(4-section_name), 'false_cell')
                    os.makedirs(directory_false, exist_ok=True)
                    
                    file_path = os.path.join(directory_false, f'{sp_name}-{section_name}-{confidence}-{time.time()}.jpg')
                    cv2.imwrite(file_path, input_crop)
                    
            
                
            else:
                negative += 1
                # os.makedirs(f'crop/{sp_name}/{section_name}/no_cell',exist_ok=True)
                # cv2.imwrite(f'crop/{sp_name}/{section_name}/no_cell/{id}.jpg',input_crop)

        return negative, positive