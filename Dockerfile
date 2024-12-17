FROM python:3.9-slim-bullseye

RUN apt-get update && apt-get install -y default-jre && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install pyspark==3.3.2 numpy

WORKDIR /app
COPY predict.py .
COPY model ./model

ENTRYPOINT ["python", "predict.py"]

