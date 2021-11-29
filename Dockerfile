FROM python:3.9.1
ADD . /simple-metric-learning
WORKDIR /simple-metric-learning
RUN pip install -r requirements.txt