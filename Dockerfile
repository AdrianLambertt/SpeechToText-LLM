FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

WORKDIR /SpeechToText-LLM
COPY . /SpeechToText-LLM

RUN pip install -r requirements.txt