FROM python:3.11-slim
WORKDIR /app
ARG INSTALL_OPTIONAL=1
COPY backend/requirements.txt backend/requirements-optional.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && if [ "$INSTALL_OPTIONAL" = "1" ]; then pip install --no-cache-dir -r requirements-optional.txt; fi
COPY backend/app ./app
COPY backend/scripts ./scripts
COPY sample_papers ./sample_papers
ENV SYNAPSE_DATA_DIR=/data SYNAPSE_SAMPLE_DIR=/app/sample_papers
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
