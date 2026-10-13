FROM python:3.11-slim

# Run as an unprivileged user instead of root
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser app.py .

# Gradio reads these variables, so app.py needs no changes
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_ANALYTICS_ENABLED=False
ENV PYTHONUNBUFFERED=1

USER appuser

EXPOSE 7860

CMD ["python", "app.py"]
