FROM python:3.11-slim

# Run as an unprivileged user instead of root
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser app.py .

# Gradio reads these variables, so app.py needs no changes
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_ANALYTICS_ENABLED=False
ENV PYTHONUNBUFFERED=1

# The hosting platform sets PORT (SnapDeploy uses 1024). 1024 is the default here
# so the container still works if PORT is not provided.
ENV PORT=1024

USER appuser

EXPOSE 1024

# Start Gradio on whatever port the platform provides
CMD ["sh", "-c", "GRADIO_SERVER_PORT=${PORT} python app.py"]
