FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model_contract.json .

RUN mkdir -p /app/model

# If deployment/model/fold4_best.pt exists in the build context,
# uncomment the following line for the simplest class-demo deployment:
# COPY model/fold4_best.pt /app/model/fold4_best.pt

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
