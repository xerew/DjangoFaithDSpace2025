FROM rasa/rasa-sdk:3.7.0
USER root
COPY requirements.txt /actions/requirements.txt
RUN pip install --no-cache-dir -r /actions/requirements.txt
USER 1001
