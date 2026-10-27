FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Mount the dataset at runtime:
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" telco-churn
ENTRYPOINT ["python", "-m", "src.train"]
