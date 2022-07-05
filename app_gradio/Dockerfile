# The "buster" flavor of the official docker Python image is based on Debian and includes common packages.
# Keep any dependencies and versions in this file aligned with the environment.yml and Makefile
FROM python:3.7-buster

# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo

# Install Python dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip==21.1.3
RUN pip install -r requirements.txt

# Copy only the relevant directories
COPY text_recognizer/ ./text_recognizer
COPY app_gradio/ ./app_gradio

# Run the web server
EXPOSE 11700
ENV PYTHONPATH /repo
ENTRYPOINT ["python3", "/repo/app_gradio/app.py", "--port", "11700"]
