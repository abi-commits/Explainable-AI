FROM python:3.9-slim

WORKDIR /app

COPY pyproject.toml .
COPY README.md .
COPY src /app/src
COPY config /app/config

# Install dependencies
RUN pip install --no-cache-dir .

# Expose Streamlit port
EXPOSE 8501

ENV PYTHONPATH=/app/src
ENV ENV=prod

CMD ["streamlit", "run", "src/explainable_aml/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
