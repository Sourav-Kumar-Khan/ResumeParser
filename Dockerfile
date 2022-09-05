FROM python:3.7.13
WORKDIR /resume
COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt
EXPOSE 8501
COPY ./*.py ./
ENTRYPOINT ["streamlit","run"]

CMD ["web.py"]


