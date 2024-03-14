import os
import cv2
import numpy as np
from skimage.filters import unsharp_mask
from skimage import img_as_ubyte
from backend.model_inference import ClassificationModel

class ImageProcessor:
    def __init__(self,model_section):
        self.section_class_name = ['no_pillar','pillar']
        self.model = ClassificationModel(model_section,self.section_class_name)
        
    def getAngleByContour(self,contours):
        maxLength=0
        maxContour=None

        #find the contour of line with maximum length
        for contour in contours:
            length=cv2.arcLength(contour, False)
            if length> maxLength:
                maxLength=length
                maxContour=contour

        # Fit a line to the contour
        vx, vy, x, y = cv2.fitLine(maxContour, cv2.DIST_L2, 0, 0.01, 0.01)

        # Calculate angle using the arctan2 function
        angle = np.arctan2(vy, vx) * 180 / np.pi

        return angle
        
    def rotate(self,image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]

        if center is None:
            center = (w / 2, h / 2)

        # Perform the rotation
        M = cv2.getRotationMatrix2D(center, float(angle), scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated

    def angle_corrections(self,image):
        # Resize the image
        originalImage = image.copy()
        image1=cv2.resize(image, (700,400))
        copyImage=image1.copy()

        # Check if the image is BGR, then convert to grayscale
        if len(copyImage.shape) == 3:
            copyImage = cv2.cvtColor(copyImage, cv2.COLOR_BGR2GRAY)

        # # Apply Otsu's thresholding
        thresh = cv2.threshold(copyImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        binary = 255 - cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        binary = cv2.medianBlur(binary, ksize=13)


        # Apply Canny edge detection
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)


        # Detect lines using the Hough Line Transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

        # Draw detected lines on the original image
        # image1 = cv2.cvtColor(copyImage, cv2.COLOR_GRAY2BGR)

        avg_angle = []
        if lines is not None:
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta)
                # Filter lines with angles close to horizontal (0 or 180 degrees)
                if 85 <= angle <= 95 :
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(image1, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    avg_angle.append(angle-90)

            final_angle = (sum(avg_angle) / len(avg_angle)) if len(avg_angle) > 0 else 0


        else:
            contours,_=cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            final_angle=self.getAngleByContour(contours)
    

        image_res = self.rotate(originalImage, final_angle)

        return image_res, final_angle


    def combine_points(self, points, threshold=300):
        combined_points = [points[0]]
 
        for point in points[1:]:
            if point - combined_points[-1] <= threshold:
                combined_points[-1] = (combined_points[-1] + point) // 2
            else:
                combined_points.append(point)
 
        return sorted(combined_points)
 
    def process_image(self, window):
 
        classification_class = self.model.classify_roi(window)
 
        if classification_class == 'no_pillar':
            return True
        return False
 
    def get_sections(self, image_path, specimen_name):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        scaling_factor = 255.0 / np.max(image)
        image = (image * scaling_factor).astype(np.uint8)
        image_channel3 = cv2.imread(image_path)
 
        scaling_factor = 255.0 / np.max(image_channel3)
        image_channel3 = (image_channel3 * scaling_factor).astype(np.uint8)
 
        name = os.path.splitext(os.path.basename(image_path))[0]
       
        database_specimen_path = os.path.join(os.getcwd(),'database',f'{specimen_name}','sections_images')
        os.makedirs(database_specimen_path,exist_ok=True)
 
 
        ##################### angle correction #################
        image,final_angle = self.angle_corrections(image)
 
        if len(image.shape) > 2 and image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
        window_size = 100
        stride = 100
        lines = []
        height, width = image.shape
        mid_y = height // 2
        crops = []
 
        for x_start in range(0, width, stride):
            x_end = x_start + window_size
            if x_end > width:
                break
 
            window = image_channel3[mid_y - window_size // 2: mid_y + window_size // 2, x_start: x_end]
 
            if self.process_image(window):
                lines.append(x_start + window_size // 2)
 
        sorted_lines = self.combine_points(lines)
 
        for id, line in enumerate(sorted_lines[1:], start=1):
            image_channel3_plotted = cv2.line(image_channel3, (sorted_lines[id - 1], 0), (sorted_lines[id - 1], image.shape[0]),
                                  (255, 0, 255), thickness=10, lineType=cv2.LINE_8, shift=0)
 
            crop = image[:, max(sorted_lines[id - 1] - 100, 0):line + 100]
            if crop.shape[1] > 4000:
                crop = unsharp_mask(crop, radius=5, amount=2)
                crop = img_as_ubyte(crop)
 
                ### triming section
                crop = crop[400:-400,90:crop.shape[1]-90]
 
                if crop.shape[1] > 5000:
                    crop = crop[:,:4700]
               
 
                curr_dir = os.getcwd()
    
                crop_path = os.path.join(curr_dir,database_specimen_path,f'{5-id}.png')
               
                crops.append(crop_path)
                cv2.imwrite(crop_path, crop)
 
 
        return image_channel3_plotted, crops