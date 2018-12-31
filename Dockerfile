FROM python:3.6-slim-stretch

RUN apt-get update && apt-get install -y python3-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app app/

RUN python app/server.py

EXPOSE 5042

CMD ["python", "app/server.py", "serve"]