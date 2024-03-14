
import cv2
import os
import glob 
import numpy as np 
from skimage.feature import match_template
from skimage.feature import peak_local_max
from backend.model_inference import ClassificationModel
import time 

class PillarDetector:
    def __init__(self, model_path, pillar_class=['positive_pillar','negative_pillar']):
        self.model_pillar_filter = ClassificationModel(model_path, pillar_class)

    @staticmethod
    def combine_points(points, threshold=300):
        combined_points = [points[0]]

        for i in range(1, len(points)):
            if points[i] - combined_points[-1] <= threshold:
                combined_points[-1] = (combined_points[-1] + points[i]) // 2
            else:
                combined_points.append(points[i])

        return sorted(combined_points)

    @staticmethod
    def check_and_convert_to_bgr(image):
        # Check if the image is grayscale
        if len(image.shape) == 2:
            # Convert grayscale image to BGR
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            return image

    @staticmethod
    def convert_to_grayscale(image):
        # Check if the image is grayscale
        if len(image.shape) == 2 or image.shape[2] == 1:
            # Image is already grayscale
            return image
        else:
            # Convert BGR image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return gray_image

    # def process_cont(self, image_section, sec_id, sp_name):
    #     file_name = os.path.join('patches_database', f'section_{sec_id+1}_{sp_name}.png')
        
    #     # Check if the file exists
    #     if os.path.exists(file_name):
    #         # Read the image if the file exists
    #         return cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    #     else:
    #         print('Handle the case where the file doesnt exist')
    #         file_name = glob.glob('patches_database/*')
    #         return cv2.imread(file_name[0], cv2.IMREAD_GRAYSCALE)
        

    def process_cont(self,image_section,sec_id,sp_name):
        
        file_name = os.path.join(os.getcwd(),'patches_database', f'section_{sec_id+1}_{sp_name}.png')
        
        if os.path.isfile(file_name):
            return cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
        else:
            images_file = sorted(glob.glob(os.path.join(os.getcwd(),'patches_database','*')))
            for image_path in images_file:
                patch = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                patch_height,patch_width = patch.shape
                
                
                sample_mt = match_template(image_section, patch)
                pillars_found = peak_local_max(sample_mt, threshold_abs=0.85)
                
                if len(pillars_found)>0:
                    print('Pillars found',image_path)
                    for index,(y,x) in enumerate(pillars_found[:1]):
                        patch = image_section[y:y+patch_height,x:x+patch_width]
 
                    return patch

    def recursive_process(self, npimg, npimg_plot, patch, all_points, thresh, sp_name, sec_id, model_activation=False):
        # Your recursive_process function code here
        patch = self.convert_to_grayscale(patch)
        out_patch= patch

        
        sample_mt = match_template(npimg_plot, patch)
        pillars_found = peak_local_max(sample_mt, threshold_abs=thresh)

        # Draw bounding boxes around detected pillars
        # npimg_bgr = cv2.cvtColor(npimg, cv2.COLOR_GRAY2BGR)
        patch_height,patch_width = patch.shape


        comp_lenght = len(all_points)

        if model_activation:
            for index,(y,x) in enumerate(pillars_found):
                patch = npimg[y:y+patch_height,x:x+patch_width]

                if sec_id==0 and comp_lenght+index>9800:
                    return npimg_plot, out_patch , all_points
                elif sec_id==1 and comp_lenght+index>10900:
                    return npimg_plot, out_patch , all_points

                elif sec_id==2 and comp_lenght+index>12250:
                    return npimg_plot, out_patch , all_points
                elif sec_id==3 and comp_lenght+index>14000:
                    return npimg_plot, out_patch , all_points
                
                patch = self.check_and_convert_to_bgr(patch)
                
                classification_class = self.model_pillar_filter.classify_roi(patch)

                if classification_class=='positive_pillar':
                    cv2.rectangle(npimg_plot, (x+2,y+2), (x + patch_width-2, y + patch_height-2), (255,255,255), -1)
                    centroid_x = x + patch_width // 2
                    centroid_y = y + patch_height // 2

                    all_points.append((x,y,patch_width,patch_height))
                    # directory = os.path.join(os.getcwd(),'database', sp_name,'occlusion_images', 'cell')
                    # os.makedirs(directory, exist_ok=True)

                    # file_path = os.path.join(directory, f'{time.time()}.jpg')
                    # cv2.imwrite(file_path, patch)
                    out_patch=patch
                # else:
                #     os.makedirs(f'all_crops/pillars/{sp_name}/{max_prob_class}', exist_ok=True)
                #     cv2.imwrite(f'all_crops/pillars/{sp_name}/{max_prob_class}/{time.time()}.jpg', patch)
        else:
            for index,(y,x) in enumerate(pillars_found):
                patch = npimg[y:y+patch_height,x:x+patch_width]
                cv2.rectangle(npimg_plot, (x+2,y+2), (x + patch_width-2, y + patch_height-2), (255,255,255), -1)
                centroid_x = x + patch_width // 2
                centroid_y = y + patch_height // 2

                all_points.append((x,y,patch_width,patch_height))
            
                # os.makedirs(f'all_crops/pillars/{sp_name}/positive_pillar', exist_ok=True)
                #cv2.imwrite(f'all_crops/pillars/{sp_name}/positive_pillar/{time.time()}.jpg', patch)
                out_patch=patch

        return npimg_plot, out_patch , all_points

    def detect_pillars(self, npimg, patch, sec_id, specimen_name):
        root_patch = patch
        npimg_plot = npimg.copy()
        all_points = []

        database_section_plotted_path = os.path.join(os.getcwd(),'database', f'{specimen_name}','section_plotted_images')
        os.makedirs(database_section_plotted_path,exist_ok=True)

        for st, thresh in enumerate([0.75, 0.78, 0.75, 0.75, 0.73, 0.70, 0.65, 0.68, 0.65, 0.63, 0.60, 50, 0.50, 0.48]):
            if st > 0:
                model_activation = True
            else:
                model_activation = False

            if st == 4:
                old_patch = patch

            if st > 5:
                npimg_plot, patch, all_points = self.recursive_process(npimg, npimg_plot, old_patch, all_points, thresh, specimen_name, sec_id, model_activation)
                old_patch = patch

            npimg_plot, patch, all_points = self.recursive_process(npimg, npimg_plot, patch, all_points, thresh, specimen_name, sec_id, model_activation)

            if sec_id == 0 and len(all_points) > 9850:
                all_points = all_points[:9850]
                plotted_image_path = os.path.join(database_section_plotted_path,f'{4-sec_id}.jpg')
                cv2.imwrite(plotted_image_path, npimg_plot)
                return npimg, len(all_points), all_points,plotted_image_path
            
            elif sec_id == 1 and len(all_points) > 10900:
                all_points = all_points[:10900]
                plotted_image_path = os.path.join(database_section_plotted_path,f'{4-sec_id}.jpg')
                cv2.imwrite(plotted_image_path, npimg_plot)
                return npimg, len(all_points), all_points,plotted_image_path
            
            elif sec_id == 2 and len(all_points) > 12300:
                all_points = all_points[:12300]
                plotted_image_path = os.path.join(database_section_plotted_path,f'{4-sec_id}.jpg')
                cv2.imwrite(plotted_image_path, npimg_plot)
                return npimg, len(all_points), all_points ,plotted_image_path
            
            elif sec_id == 3 and len(all_points) > 14020:
                all_points = all_points[:14020]
                plotted_image_path = os.path.join(database_section_plotted_path,f'{4-sec_id}.jpg')
                cv2.imwrite(plotted_image_path, npimg_plot)
                return npimg, len(all_points), all_points ,plotted_image_path
            
        plotted_image_path = os.path.join(database_section_plotted_path,f'{4-sec_id}.jpg')
        cv2.imwrite(plotted_image_path, npimg_plot)
        return npimg, len(all_points), all_points,plotted_image_path
