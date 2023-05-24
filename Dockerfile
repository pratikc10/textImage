# Use the official lightweight Python image.
# https://hub.docker.com/_/python


FROM python:3.10-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# install dependencies
RUN pip install --upgrade pip 
COPY ./requirements.txt .
RUN pip install -r requirements.txt

#copy project 
COPY . .

EXPOSE 5000

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
