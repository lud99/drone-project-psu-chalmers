FROM python:3.11-slim

WORKDIR /app

COPY mavlink_client/requirements.txt /app/mavlink_client/requirements.txt
RUN pip install --no-cache-dir -r /app/mavlink_client/requirements.txt

COPY mavlink_client /app/mavlink_client

EXPOSE 8010

CMD ["python", "-m", "mavlink_client.main"]
