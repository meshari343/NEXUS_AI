FROM python:3.10.4-slim-bullseye

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
build-essential gcc 

RUN apt install git -y

WORKDIR /NEXUS_AI

COPY requirements.txt .

# RUN python3 -m pip install -U pip

RUN pip3 install -r requirements.txt

RUN python3 -m spacy download en_core_web_sm 

COPY . .

RUN pip3 install .

EXPOSE 8000

CMD ["python3", "main.py"]