import streamlit as st
from reseme_parser import *
import os
import json


st.header("Resume Parser")
st.markdown("**Upload Your Resume and get the Extact out of it.**")

st.sidebar.subheader("Upload your Resume here :-")
file = st.sidebar.file_uploader('Choose a file')
if file:
    if file.name.split('.')[-1]=='pdf':
     
        text = extract_text_from_pdf(file)
        st.subheader("Here what we got from your resume")
        st.write(text)
    elif file.name.split('.')[-1]=='doc' or file.name.split('.')[-1]=='docx':
        text = extract_text_from_docx(file)
        st.subheader("Here what we got from your resume")
        st.write(text)

    name=extract_name(text)
    contact_no= extract_phone_number(text)
    mobile_no = resume_parser(file)['mobile_number']
    email = extract_email(text)
    dob = extract_dob(text)
    skills = extract_skills(text)
    education = extract_education(text)
    college_name = resume_parser(file)['college_name']
    designation = resume_parser(file)['designation']
    experience = resume_parser(file)['experience']
    comapany_name = resume_parser(file)['company_names']
    no_of_pages = resume_parser(file)['no_of_pages']
    total_experience = resume_parser(file)['total_experience']

    data = {}
    if name is not None:
        data['name'] = name
    if contact_no is not None:
        data['Contact no'] = list(contact_no)
    if mobile_no is not None:
        data['mobile no'] = mobile_no
    if email is not None:
        data['email'] = list(email)
    if dob is not None:
        data['dob'] = dob
    if skills is not None:
        data['skills'] = list(skills)
    if education is not None:
        data['education'] =list(education)
    if college_name is not None:
        data['college names'] = college_name
    if designation is not None:
        data['designation'] = designation
    if experience is not None:
        data['experience'] = experience
    if comapany_name is not None:
        data['company names'] = list(comapany_name)
    data['no of pages'] = no_of_pages
    if total_experience is not None:
        data['total experience'] = total_experience


    st.markdown("\n")
    st.markdown("**The json file we save like this**")
    st.markdown("\n")
    st.json(data, expanded=True)
    #save the json file
    file_name = file.name.split(".")[0]+".json"
    with open(file_name, 'w') as fp:
        json.dump(data, fp,indent=4)
             
     
     
        
