FROM python:3.8-slim

ENV PYTHONUNBUFFERED 1

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y