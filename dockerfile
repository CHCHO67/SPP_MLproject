FROM python:3.8

RUN apt-get -y update

RUN pip install --upgrade pip

COPY requirements.txt /tmp/
COPY Stock_Price_Prediction.py /app/

RUN pip install --upgrade certifi

RUN pip install --no-cache-dir -r /tmp/requirements.txt