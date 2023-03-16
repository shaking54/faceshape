# syntax=docker/dockerfile:1
# FROM ubuntu:latest
FROM python:3.7

RUN apt-get update && \
    apt-get -y install cmake protobuf-compiler && \
    apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /code

COPY requirements.txt requirements.txt

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["tail"]
CMD ["-f","/dev/null"]
