FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN  pip install -U pip 

RUN  pip install -r requirements.txt 

COPY api/ ./api

COPY src/ ./src

COPY models/ ./models

COPY dataset/ ./dataset

COPY models/random_forest.joblib ./model/random_forest.joblib

COPY initializer.sh .

RUN chmod +x initializer.sh

EXPOSE 8000

ENTRYPOINT ["./initializer.sh"]
