FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8386

ENV MODEL_PATH=/app/models/model_diabetes_v0.1.pkl
ENV SCALER_PATH=/app/models/scaler_diabetes_v0.1.pkl

CMD ["uvicorn", "scripts.api_v0.1:app", "--host", "0.0.0.0", "--port", "8386"]
