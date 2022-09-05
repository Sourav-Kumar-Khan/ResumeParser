FROM python:3.7.13
WORKDIR /resume
COPY . /resume
RUN pip3 install -r requirements.txt
#RUN pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
#RUN streamlit run web.py
EXPOSE 8501
#ENTRYPOINT ['streamlit','run']
CMD streamlit run web.py
