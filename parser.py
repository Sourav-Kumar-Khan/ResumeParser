from fastapi import FastAPI,File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import boto3
import requests
import pandas as pd
import json
import uuid
import shutil
import numpy as np
from config import CONFIG
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import concurrent.futures
import re
import openai
from tqdm import tqdm
#virtual env yolo_api
from typing import Union
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# to get a string like this run:
##################################3 openssl rand -hex 32
security = HTTPBearer()

TOKEN = CONFIG['SECURITY TOKEN']
app = FastAPI()
 
token = CONFIG['OPENAI TOKEN']
openai.api_key = token
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
   # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#import pickle
#model = pickle.load(open('model.pkl', 'rb'))
#transform = pickle.load(open('transform.pkl', 'rb'))
##get the json of the image file
def get_json(file_path):
    ACCESS_KEY_ID = CONFIG['ACCESS_KEY_ID']
    ACCESS_SECRET_KEY = CONFIG['ACCESS_SECRET_KEY']



    textractclient = boto3.client("textract", aws_access_key_id=ACCESS_KEY_ID,
                                aws_secret_access_key=ACCESS_SECRET_KEY, region_name="us-east-2")
    with open(file_path, "rb") as f:
        png_encoded = f.read()                           
     
    response = textractclient.detect_document_text(
        Document={
            'Bytes': png_encoded
        }
    )
    return response["Blocks"]
#To get the dataFrame structure of the json response
def get_dataframe(json_file):
    Left=[]
    Top = []
    Text = []
    for i in range(1,len(json_file)):
        try:
            left = json_file[i]['Geometry']['BoundingBox']['Left']
            top = json_file[i]['Geometry']['BoundingBox']['Top']
            text =json_file[i]['Text']
            Left.append(left)
            Top.append(top)
            Text.append(text)
        except:
            print(i)
            continue
    d  = {'left':Left, 'top':Top, 'text':Text}
    df = pd.DataFrame(data = d)
    return df

def get_bbox(path):
    
    os.system(f'python yolov5/detect.py --source {path} --weights weights/last.pt --img 416 --save-txt --save-conf')
########################################################
def get_info(edu):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt= "What is the full name, job position, address/location and years of experience in the resume segement?.Return all the deteils in a json object.\n"+"Education:\n"+ edu,
        temperature=0,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    output_text = response['choices'][0]['text']
    output_text = output_text.replace("\n","")
    try:
        start = output_text.find("{",)
        end = output_text.find("}")
        output_text = output_text[start:end+1]
        return json.loads(output_text)
    except:
        return output_text
#################################3
import docx2txt
def extract_text_from_docx(docx_path):
    txt = docx2txt.process(docx_path)
    if txt:
        return txt.replace('\t', ' ')
    return None

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)
# Extract Phone Number
import re
def get_linkedine(text):
    # Define the regular expression pattern
    pattern = r"linkedin\.com/in/[a-zA-Z0-9]+"

    # Use the re.search function to extract the match
    match = re.search(pattern,text)

    # Extract the matched string
    if match:
        matched_string = match.group()
        return matched_string
    else:
        return "No Match Found."
def get_github(text):
    
    # Define the pattern to match
    pattern = re.compile(r"github\.com/([a-zA-Z0-9]+)")

    # Search for the pattern in the URL
    match = pattern.search(text)

    # Extract the username if the pattern was found
    if match:
        username = match.group()
        return username
    else:
        return "No Match Found"
        
        
def extract_phone_number(resume_text):
     
    PHONE_REG = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
    phone = re.findall(PHONE_REG, resume_text)
    try:
        return phone[0]
    except:
        return "NA"

# Email Address
def extract_email(text):
    EMAIL_REG = re.compile(r'[a-zA-Z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+')
    return re.findall(EMAIL_REG, text)

########### experience  & education
def get_class_txt(dc,className):
    lls = ""
    for l in range(len(dc)):
        for i in range(len(dc[l]['class'])):
            if dc[l]['class'][i]['class_name'] ==className:
                lls+=dc[l]['class'][i]['text']
    return lls
###############experience

# jp_list = []
# with open(os.path.join(os.getcwd(),"resource/job_position.txt"), 'r') as f:
#     jp_list.append(f.readlines())
# jp_list =  [i.replace('\n', '') for i in jp_list[0]]

def get_position(context,jp_list):
    jp_score =[]
    for j in jp_list:
        match = {
                "questions": [
                    context,
                ],
                "resume_summ": j,
                "extra_res": "Hey"
                }
        r = requests.post("https://similarity.sproutsai.com/sim", data = json.dumps(match))
        jp_score.append(r.json()["similar_values"][0])
    return jp_list[np.where(jp_score>=0.6)[0].tolist()[0]]
#give a input chance to add exp question and job position
def parser_exp(exp):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt= "What is the company name, job position, the skills used, and duration in the company given in the resume. Return it as a json object.\n"+"Resume:\n"+ exp,
        temperature=0,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
        )
    output_text = response['choices'][0]['text']
    output_text = output_text.replace("\n","")
    try:
        start = output_text.find("{",)
        end = output_text.find("}")
        output_text = output_text[start:end+1]
        json_return =  json.loads(output_text)
        json_return["description"] = exp.replace("\n"," ")
        #format the experience section
        ini_list = ['Company Name', 'Job Position','Skills Used','Duration','Description']
        final_dict = dict(zip(ini_list, list(json_return.values())))
        return final_dict
    
    except:
        return output_text
def get_exp_parser(text):
    temp_dict = {}
    temp_lst = []
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
      {"role": "user", "content":"Parse the experience section of the resume:"+text+" and return strictly on the  given standard dict format: Company_Name, Job_Position, Skills_Used, Duration, Description"}
    ]
      
    )
    output_text = response['choices'][0]['message']['content']
    output_text = output_text.replace("\n"," ")
    exp = output_text.split(":")
    for i in range(1,len(exp)):
        if i<len(exp):
            key = exp[i-1].split()[-1]
            if key in "Company_Name, Job_Position, Skills_Used, Duration, Description":
                temp_dict[key] = " ".join(exp[i].split()[:-1])
    
        
        if i%5==0 and i!=1:
            standard_keys = ["Company Name", "Job Position", "Skills Used", "Duration", "Description"]
            temp_dict1 = dict(zip(standard_keys,list(temp_dict.values())))
            temp_lst.append(temp_dict1.copy())
            temp_dict.clear()  
    return temp_lst 
    
##############################
def get_answer(question, context):
    url = "http://classifier.sproutsai.com:8000/get_answer"

    payload = {
        "ques": question,
        "context": context
    }

    #headers = {'Content-type': 'application/json'}
    response = requests.post(url, data=json.dumps(payload))
    ans= response.json()['Ans']
    return ans
def get_edu(edu):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt= "What is the degree, duration, institute name and score/cgpa given in the resume segement?.Return all the deteils in a json object.\n"+"Education:\n"+ edu,
        temperature=0,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    output_text = response['choices'][0]['text']
    output_text = output_text.replace("\n","")
    try:
        start = output_text.find("{",)
        end = output_text.find("}")
        output_text = output_text[start:end+1]
        Edu =  json.loads(output_text)
        ini_list = ['Degree', 'Duration','Institute','Score/CGPA']
        Edu_dict = dict(zip(ini_list, list(Edu.values())))
        Edu_dict['Description'] = " ".join(list(Edu.values()))
        return Edu_dict
    except:
        return output_text
#####################################
#skills parse
try:
  with open('all_skill_5.json','r') as f:
      skills = json.load(f)['Data Engineer']['skills']
except Exception as e:
  print(e)
    
    
def return_skill_criteria(jd):
    criteria_dict = {}
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt= "What are the technical skills and techinical frameworks need for this resume.Return it as a json object.\n"+"Resume:\n"+ jd,
      temperature=0,
      max_tokens=250,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    output_text = response['choices'][0]['text']
    output_text = output_text.replace("\n","")
    try:
        start = output_text.find("{",)
        end = output_text.find("}")
        output_text = output_text[start:end+1]
        skills_output = json.loads(output_text)
    except:
        skills_output = output_text
    
    criteria_dict['Skills'] = skills_output      
     
    #criteria_dict['Job_Description'] = jd
    for skill in skills:
        if skill in jd.lower():
            if skills[skill] not in criteria_dict:
                criteria_dict[skills[skill]] = [skill]
            else:
                criteria_dict[skills[skill]].append(skill)
    
    lss = []
    for v in criteria_dict:
        temp_dict={}
        temp_dict['criteria'] = v
        temp_dict['skills'] = criteria_dict[v]
        lss.append(temp_dict)
    
    return lss
    
#########################333

def pdf_img(path):  
    images = convert_from_path(pdf_path=path, dpi = 100)
    files = []
    for i, image in enumerate(images):
        dir_name = os.path.join('pdf_files', path.split("/")[-1].split(".")[0])
        #print(dir_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
         
        fname = f'{dir_name}/image'+str(i)+'.png'
        image.save(fname, "PNG")
        #print("Path is :-",path.split(".")[0])
        files.append(fname)
    return files

def get_segment_text_thread(position,df,structure):
    text_data = ""
    #dc1['class']['class_id']= uuid.uuid4()
    yolo_class = ["achievements", "education","experience","personal information","skills","summary"]
    c,x_center, y_center, width, height, _= position.split()
    x,y,w,h = float(x_center), float(y_center), float(width),float(height)
    lst=[]
    for i in range(len(df)):
        l,r,t,b = (x-(w/2)),(x+(w/2)),(y-(h/2)),(y+(h/2))
        if (df.left[i]>= l and df.left[i]<= r and df.top[i]>= t and df.top[i]<=b):
            #print(i,df.left[i],df.top[i])
            for j in df.text[i].split():
                if j not in lst:
                    lst.append(j)
            #lst.append(df.text[i])
            #print(df.text[i])
            text_data = " ".join(lst)

    # lst56.append({"class_name": cls1[model.predict(transform.transform([text_data]).toarray())[0]] , "text":text_data})
    ##############
    # url = f'http://classifier.sproutsai.com:8000/classify?sequence_to_classify={text_data}'
    # response = requests.post(url)
    # r = response.json()['label']
    r = yolo_class[int(c)]
    if r=="experience":
        structure['experience'].append(parser_exp(text_data))
            
    #if r in "education":
     #   structure['Education']['Description'] = text_data
        #edu_data.append(edu_parse(text_data))
            
    return {"class_name": r , "text":text_data}


####get segment from text
def get_text_from_segment(file_path,df,structure):
    try:
        with open(file_path,'r') as f:
            txt = f.readlines()
    except:
        return {"msg":"this page is blank"}
    # yolo_class = ["achievements", "education","experience","personal information","skills","summary"]
    # classifier_class = ['contact_about_me', 'education', 'personal_projects_certification','skills', 'work_experience']
   
    dc1 = {}
    dc1['id'] = uuid.uuid4()
    #dc1['class']  = {}
    lst56 =[]
    try:
        with ThreadPoolExecutor() as executor:
            results = [executor.submit(get_segment_text_thread,txt_data,df,structure) for txt_data in txt]
        
            for f in concurrent.futures.as_completed(results):
                lst56.append(f.result())
    except:
        lst56.append([get_segment_text_thread(txt_data,df,structure) for txt_data in txt])
    
    
    return lst56

def multiprocess_func(text,pdf_file_path):
    executors_list = []
    r_list = []

    with ThreadPoolExecutor() as executor:
        executors_list.append(executor.submit(get_info, text))
        executors_list.append(executor.submit(extract_email, text)) 
        executors_list.append(executor.submit(extract_phone_number, text))
        executors_list.append(executor.submit(get_github, text)) 
        executors_list.append(executor.submit(get_linkedine, text))
        executors_list.append(executor.submit(return_skill_criteria, text))
        #executors_list.append(executor.submit(pdf_img,pdf_file_path))
        executors_list.append(executor.submit(get_edu,text))
        executors_list.append(executor.submit(get_exp_parser,text))

    for x in executors_list:
        r_list.append(x.result())
    return r_list
#######
@app.post("/image/")
async def process_image(file: UploadFile = File(...), credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    if credentials.scheme != "Bearer":
        raise HTTPException(status_code=400, detail="Invalid authentication scheme")
    if credentials.credentials != TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    #delete the path
    #shutil.rmtree(os.path.join('yolov','runs','detect'), ignore_errors=False)
    #save the image file
    if file.content_type != 'image/png':
        raise HTTPException(status_code=400, detail="File should be in png format.")
    file_path = "image"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with open(f"{file_path}/{file.filename}", "wb") as f:
        f.write(await file.read())
    #detect the segment with the yolo api
    os.system(f'python yolov5/detect.py --source {f"{file_path}/{file.filename}"} --weights weights/last.pt --img 416 --save-txt --save-conf')
    os.remove(f"{file_path}/{file.filename}")
    #read the text file of coordinates
    with open(os.path.join("yolov5","runs","detect","exp","labels",f"{file.filename.replace('png','txt')}"),'r') as f:
        txt = f.readlines()
    shutil.rmtree(os.path.join('yolov5','runs','detect'), ignore_errors=False)
    return {"file_name": file.filename,"coordinates": txt}

 
@app.post("/resume_parser",tags=['resume parser'])
async def upload(file: UploadFile = File(...), credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict :
    if credentials.scheme != "Bearer":
        raise HTTPException(status_code=400, detail="Invalid authentication scheme")
    if credentials.credentials != TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    structure = {}
    shutil.rmtree('pdf_files', ignore_errors=True)
    os.makedirs('pdf_files')
    shutil.rmtree("yolov5/runs/detect", ignore_errors=True)
    if file.filename.rsplit(".",1)[-1] == "pdf":
        try:
            contents = file.file.read()
            pdf_file_path = os.path.join("pdf_files",str(uuid.uuid4())+".pdf")
            with open(pdf_file_path, 'wb') as f:
                f.write(contents)
        except Exception as e:
            return {"message": "There was an error uploading the file","Error":e}
        finally:
            file.file.close()
        # pdf_file_path = os.path.join("pdf_files",file.filename)
        # print(pdf_file_path)
        os.chmod(pdf_file_path, 0o777)
        #pdf is getting parse into images and save into a file.filename directory
        #parse the images in a loop
        # list_img = [files for files in os.listdir(dir_name_img)]
        text = extract_text_from_pdf(pdf_file_path).replace("\n"," ")
    elif file.filename.rsplit(".",1)[-1] == "docx":
        try:
            contents = file.file.read()
            pdf_file_path = os.path.join("pdf_files",str(uuid.uuid4())+".docx")
            with open(pdf_file_path, 'wb') as f:
                f.write(contents)
        except Exception as e:
            return {"message": "There was an error uploading the file","Error":e}
        finally:
            file.file.close()
        text = extract_text_from_docx(pdf_file_path)
    else:
        return {"msg": "Please provide the resume in pdf or docx format only"}
    r_list = multiprocess_func(text,pdf_file_path)
    structure['personal_info'] = r_list[0]
    try: 
        structure['personal_info']['email'] = r_list[1]
        structure['personal_info']['phone number'] = r_list[2]
        structure['personal_info']['Github Links'] = r_list[3]
        structure['personal_info']['Linkedine Links'] = r_list[4]
    except:
        structure['email'] = r_list[1]
        structure['phone number'] = r_list[2]
        structure['Github Links'] = r_list[3]
        structure['Linkedine Links'] = r_list[4]
    structure['skills'] = r_list[5]
    #dir_name_img = r_list[6]
    structure['Education'] = r_list[6]
    #structure['education'] = []
    structure['experience'] = r_list[7]
    """
    dc = []
    with ProcessPoolExecutor() as executor:
            executor.map(get_bbox,dir_name_img)
        
    for i,img_path in enumerate(dir_name_img):
        if img_path.endswith(".png"):
            # get_bbox(img_path)
            # path=os.path.join("image_upload/",file.filename)
            json_data = get_json(img_path)
                
            df = get_dataframe(json_data)
            # try:
            #     get_bbox(img_path)
            # except:
            #     return {"msg": "Their is an error from segment detection"}
            # import time
            # time.sleep(10)
            file_name = img_path.split("/")[-1].replace("png", "txt")
            path1 = "yolov5/runs/detect/exp"
            # if i==0:
            #     path1 = "yolov5/runs/detect/exp"
            # else:
            #     path1 = f"yolov5/runs/detect/exp{i+1}"
            # ex=[]
            # for files in os.listdir(path):
            #     ex.append(files)
            file_path = os.path.join(path1,'labels',file_name)
            print(file_path)
            class_data = get_text_from_segment(file_path=file_path, df=df,structure=structure)
            dc.append(class_data)

        """
        #return dc1
        
    return {"id":uuid.uuid4(),"status": f"Successfully parse {file.filename}", "data":structure, "summary":text}


# @app.get("/")
# async def read_root():
#     return {"Hello": "This is Resume Parser API "}

@app.get("/")
async def main():
    content = """
<!DOCTYPE html>
<html>
<head>
  <title>SproutsAI Resume File Uploader</title>
  <style>
    /* Set the background color */
    body {
      background-color: #F5F5F5;
    }

    /* Center the container */
    .container {
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
    }

    /* Style the file input */
    .file-input {
      padding: 10px;
      border: 2px solid #008080;
      border-radius: 5px;
      font-size: 18px;
      margin-bottom: 10px;
    }
    
    /* Style the drag-and-drop container */
    .drag-and-drop {
      background-color: #008080;
      color: white;
      padding: 50px;
      border-radius: 5px;
      text-align: center;
      cursor: pointer;
      transition: background-color 0.2s ease;
    }

    /* Change the background color on hover */
    .drag-and-drop:hover {
      background-color: #004D4D;
    }

    /* Style the upload button */
    .upload-button {
      background-color: #008080;
      color: white;
      padding: 10px 20px;
      border: none;
      border-radius: 5px;
      font-size: 18px;
      cursor: pointer;
    }

    /* Style the browse button */
    .browse-button {
      background-color: #008080;
      color: white;
      padding: 10px 20px;
      border: none;
      border-radius: 5px;
      font-size: 18px;
      cursor: pointer;
    }

    /* Change the button color on hover */
    .upload-button:hover, .browse-button:hover {
      background-color: #004D4D;
    }
  </style>
</head>
<body>
  <div class="container">
    <form action="/resume_parser" enctype="multipart/form-data" method="post">
      <div class="drag-and-drop" ondrop="uploadFile(event)" ondragover="return false">
        Drag and drop your pdf file here or
      </div>
      <br>
      <input type="file" name="file" id="file-input" class="file-input" accept=".pdf" onchange="uploadFile()">
      <br>
      <button type="button" class="browse-button" onclick="document.getElementById('file-input').click()">Browse</button>
      <button type="submit" class="upload-button">Upload</button>
    </form>
     

    """
    return HTMLResponse(content=content)

@app.post("/get_text")
def upload(file: UploadFile = File(...), credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer":
        raise HTTPException(status_code=400, detail="Invalid authentication scheme")
    if credentials.credentials != TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    try:
        contents = file.file.read()

        with open('pdf_files/'+str(file.filename), 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    pdf_file_path = os.path.join("pdf_files",file.filename)
    t = extract_text(pdf_file_path)
    t = t.replace("\n", " ")
    res = re.sub(r'[^\w\s]', '', t)
    ##collect information from the version1 
    url ="http://3.138.143.170:8000/upload"
    files = {'file': open(pdf_file_path, 'rb')}
    r = requests.post(url, files=files)
    os.remove(pdf_file_path)
    data = json.loads(r.text)
    name = data['personal_information']
    contact_details = data['contact_details']
    return {"name":name, "contact_details":contact_details,"text":res}
     


@app.post("/parse_resume_image")
async def resume_parser(file: UploadFile = File(...), credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer":
        raise HTTPException(status_code=400, detail="Invalid authentication scheme")
    if credentials.credentials != TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    try:
        contents = file.file.read()
        with open('pdf_files/'+str(file.filename), 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    path=os.path.join("pdf_files/",file.filename)
    json_data = get_json(path)
    df = get_dataframe(json_data)
    get_bbox(path)
    # import time
    # time.sleep(10)
    try:
        file_name = file.filename.replace(".png",".txt")
    except:
        file_name = file.filename.replace(".jpg",".txt")
    path1 = "yolov5/runs/detect/exp"

    file_path = os.path.join(path1,'labels',file_name)

    class_data = get_text_from_segment(file_path=file_path, df=df)

    os.remove(path)
    shutil.rmtree("yolov5/runs/detect", ignore_errors=False)
    return class_data
