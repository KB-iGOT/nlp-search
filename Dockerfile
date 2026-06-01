FROM python:3.12-slim

ENV VIRTUAL_ENV=/opt/venv

# Create a virtual environment
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Create non-root user
RUN addgroup --system appuser && adduser --system --ingroup appuser appuser

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install uv && uv pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src /app/src/

# Change ownership of app directory
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
