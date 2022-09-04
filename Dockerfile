FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD python -m spacy download en_core_web_sm
