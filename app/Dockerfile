FROM continuumio/miniconda3

WORKDIR /home/app

RUN apt-get update
RUN apt-get install nano unzip
RUN apt install curl -y
RUN apt-get update && apt-get install -y cmake
RUN apt-get update && apt-get install -y python3-opencv
RUN conda install -c conda-forge dlib
RUN pip install opencv-python
RUN pip install --upgrade pip
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt

COPY . /home/app

CMD streamlit run --server.port $PORT app.py