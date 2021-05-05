# syntax=docker/dockerfile:1
FROM python:3.8-slim-buster

WORKDIR /rex
RUN mkdir -p /rex

# Copy package
COPY . /rex

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["rex"]
