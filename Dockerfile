FROM python:3.7
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
CMD ['streamlit run','./app/web.py']
