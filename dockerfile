FROM python:3.8

RUN apt-get -y update

RUN pip install --upgrade pip

# 미리 작성한 requirements.txt 
COPY requirements.txt /tmp/

COPY Stock_Price_Prediction.py /app/

RUN pip install --no-cache-dir -r /tmp/requirements.txt