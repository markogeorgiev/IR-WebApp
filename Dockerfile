# syntax=docker/dockerfile:1.5
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY . .

ENV FLASK_APP=app.py
EXPOSE 8000

CMD ["python", "app.py"]
