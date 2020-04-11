FROM python:3.7

RUN pip install torch
RUN pip install future
RUN pip install numpy
RUN pip install sklearn
RUN pip install networkx
RUN apt-get update
RUN apt-get install vim -y

COPY . /notebooks