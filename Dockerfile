FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data/logs
RUN mkdir -p data/models

# Train the ML model during build
RUN python test_ml_model.py || echo "Model training failed - continuing without model"

CMD ["python", "main.py"]