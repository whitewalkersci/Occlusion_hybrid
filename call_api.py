import requests
import time 

# url = 'http://127.0.0.1:8000/process_image'
# image_path = '/home/whitewalker/Desktop/Sciverse_2/Occlusion/assets/merged/Specimen021.tif'
# payload = {'image_path': image_path}
# response = requests.post(url, json=payload)
# print(response)
# print(response.json())


input_data = {
    'specimen_name': 'Specimen021',
    'project_id': 1,
    'sections_image_path': [
        'database/Specimen021/sections_images/4.png',
        'database/Specimen021/sections_images/3.png',
        'database/Specimen021/sections_images/2.png',
        'database/Specimen021/sections_images/1.png'
    ]
}
url_2 = 'http://127.0.0.1:8002/process_analysis'
payload_2 = input_data
in_time = time.time()
response_2 = requests.post(url_2, json=payload_2)


print(f'processing time taken:  {time.time()-in_time}')
print(response_2.json())



# url = 'http://127.0.0.1:5000/result'
# input_data = {'project_key':1}
# response = requests.post(url,json=input_data)
# print(response.json())