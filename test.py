from backend.section_process import ImageProcessor


image_path = '/home/whitewalker/Desktop/Sciverse_2/Occlusion_project/final_pipeline/assets/merged/Specimen_001.tif'
image_processor = ImageProcessor(model_section="/home/whitewalker/Desktop/runs/classify/train/weights/best.pt")
print(image_path)
sp_name = image_path.split('/')[-1].split('.')[0]
image, crops = image_processor.get_sections(image_path, sp_name)