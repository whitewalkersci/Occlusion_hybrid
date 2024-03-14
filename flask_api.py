from flask import Flask, request
from flask_restful import Api, Resource
import cv2
from backend.section_process import ImageProcessor
from backend.process import PillarDetector
from backend.occlusion_process import OcclusionModel
import os 
from pathlib import Path
import sqlite3
import json 
from datetime import datetime
from flask import request
from werkzeug.utils import secure_filename
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import requests
import time 
import traceback
# datetime object containing current date and time
now = datetime.now()

app = Flask(__name__)
api = Api(app)

# Azure Blob Storage configuration
connection_string = 'XYZ'
container_name = "XYZ"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)


# url_server = 'http://20.219.250.242:8005/process_analysis'
url_server = 'http://20.197.14.166:8100/process_analysis'


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','tif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_files(file_paths,project_id):
    success_count = 0
    for file_path in file_paths:
        filename = secure_filename(os.path.basename(file_path))
        if allowed_file(filename):
            # Upload the file to Azure Blob Storage
            blob_client = container_client.get_blob_client(f'{project_id}/section_images/{filename}')
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, blob_type="BlockBlob")
            success_count += 1
            print(f"Uploaded {success_count}")
    return success_count

class SectionProcessing(Resource):
    def post(self):
        try:
            data = request.get_json()
            image_path = data.get('image_path')
            
            if not image_path:
                return {'message': 'No image path provided'}, 400

            image_filename = os.path.basename(image_path)
            if not image_filename.endswith(".tif"):
                return {'message': 'Invalid image path. Image filename should start with "Specimen_" and have a .tif extension'}, 400

            model_section_path = os.path.join(os.getcwd(),'models', 'section.onnx')
            image_processor = ImageProcessor(model_section_path)

            specimen_name = image_filename.split('.')[0]
            image, crops = image_processor.get_sections(image_path, specimen_name)

            return {'message': 'Image processed successfully','specimen_name':specimen_name, 'sections_image_path': crops}, 200
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'message': f'Error processing image: {str(e)}'}, 500

class CountProcessing(Resource):
    def post(self):
        try:        
            data = request.get_json()
            specimen_name = data['specimen_name']

            sections_image_path = data['sections_image_path']  
            print("INNN")

            
            in_process = time.time()
            
            database_filename = os.path.join(os.getcwd(),'database', 'final_databases.json')


            if not os.path.exists(database_filename):
                with open(database_filename, 'w') as file:
                    json.dump({}, file)
    

            with open(database_filename, 'r') as file:
                all_data = json.load(file)
                total_elements = sum(len(value) if isinstance(value, list) else 1 for value in all_data.values())
            
            #############################################################

            # success_count = upload_files(sections_image_path,total_elements+1)   ### upload in container
            
            
            # print('succefully uploaded sections images',success_count) 
            
            temp ={}
            temp[specimen_name] ={}
        
            for i in range(1,5):
                print(f'process strated section {i}')    
                input_data = {
                        'specimen_name': specimen_name,
                        'project_id': total_elements+1,
                        'section_id':i
                    }
                
                response = requests.post(url_server, json=input_data)
                print(response.json())
                result = response.json()
                temp[specimen_name][f'section{i}'] = result
                time.sleep(10)
            

            ################################################################
            all_data[total_elements+1] = temp
            with open(database_filename, 'w') as file:
                 json.dump(all_data, file, indent=4)
                
                
            print('succefully process completed time taken' , time.time() - in_process)

            return result , 200
        except Exception as e:
            
            traceback.print_exc()
            return {'message': f'Error processing image: {str(e)}'}, 500
        
class HistoryPage(Resource):
    def get(self):
        try:
            database_json_path = os.path.join(os.getcwd(),'database', 'final_databases.json')
            with open(database_json_path, 'r') as file:
                data_from_file = json.load(file)    
                final_res = []
                
            for key, value in data_from_file.items():
                print("Key:", key)
                for specimen_key, specimen_value in value.items():
                    final_res.append({'project_name':specimen_key,'Date_time':value[specimen_key]['Date_time'],'project_id':key,'Occlusion_folder_path':os.path.join(os.getcwd(),'database',specimen_key)})
 
            return final_res , 200
        except Exception as e:
            return {'message': f'Error reading {str(e)}'}, 500
        
class ViewResult(Resource):
    def post(self):
        try:
            data = request.get_json()
            project_key = data['project_key']
            specimen_name = data['Specimen_name']
            print(project_key)

            return {os.path.join(os.getcwd(),'database',specimen_name,'occlusion_images','cell')} , 200
        except:
            return {'message': f'Error reading {str(e)}'}, 500

def get_db():
    if not os.path.exists('database.db'):
        conn = sqlite3.connect('database.db')
        conn.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    return sqlite3.connect('database.db')

class ResultShow(Resource):
    def post(self):
        try:
            data = request.get_json()
            project_key = data['project_key']
            print(project_key)
            
            database_json_path = os.path.join(os.getcwd(),'database', 'final_databases.json')
            with open(database_json_path, 'r') as file:
                data_from_file = json.load(file)
 
                
            for key, value in data_from_file.items():
                if str(key) == str(project_key):
                    for specimen_key, specimen_value in value.items():
                        final_res={'project_name':specimen_key,
                                        'Date_time':value[specimen_key]['Date_time'],
                                        'Total_occlusion_count':value[specimen_key]['Total_occlusion_count'],
                                        'Occlusion_index':value[specimen_key]['Occlusion_index'],
                                        'Section1':value[specimen_key]['Section1'],
                                        'Section2':value[specimen_key]['Section2'],
                                        'Section3':value[specimen_key]['Section3'],
                                        'Section4':value[specimen_key]['Section4'],
                                        'Section1_image': os.path.join(os.getcwd(),'database', specimen_key ,'sections_images','1.png'),
                                        'Section2_image': os.path.join(os.getcwd(),'database',specimen_key ,'sections_images','2.png'),
                                        'Section3_image': os.path.join(os.getcwd(),'database',specimen_key ,'sections_images','3.png'),
                                        'Section4_image': os.path.join(os.getcwd(),'database',specimen_key ,'sections_images','4.png'),
                                        }
                        return final_res , 200
 
            return 'No data in database' , 200
        except Exception as e:
            return {'message': f'Error reading {str(e)}'}, 500
        

class Login(Resource):
    def post(self):
        try:
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')

            with get_db() as conn:
                cur = conn.cursor()

                cur.execute("SELECT * FROM users WHERE username = ?", (username,))
                user = cur.fetchone()

                if user:
                    if password == user[2]:
                        return {'message': 'Login successful'}, 200
                    else:
                        return {'message': 'Invalid username or password'}, 401
                else:
                    return {'message': 'User does not exist. Please create a new user.'}, 404
        except Exception as e:
            return {'message': f'Error logging in: {str(e)}'}, 500

class CreateUser(Resource):
    def post(self):
        try:
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')

            with get_db() as conn:
                cur = conn.cursor()

                cur.execute("SELECT * FROM users WHERE username = ?", (username,))
                existing_user = cur.fetchone()

                if existing_user:
                    return {'message': 'Username already exists. Please choose a different username.'}, 400
                else:
                    cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                    conn.commit()
                    return {'message': 'User created successfully'}, 201
        except Exception as e:
            return {'message': f'Error creating user: {str(e)}'}, 500

api.add_resource(Login, '/login')
api.add_resource(CreateUser, '/create_user')

api.add_resource(SectionProcessing, '/process_image')
api.add_resource(CountProcessing,'/process_analysis')
api.add_resource(HistoryPage, '/history')
api.add_resource(ResultShow, '/result')

if __name__ == '__main__':
    app.run(debug=True,port=8002)
