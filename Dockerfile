FROM python:2

RUN pip install torch
RUN pip install future
RUN pip install numpy
RUN pip install sklearn
RUN apt-get update
RUN apt-get install vim -y

COPY . /notebooks