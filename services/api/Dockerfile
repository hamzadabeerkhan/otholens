FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN addgroup --system ortholens && adduser --system --ingroup ortholens ortholens

FROM base AS runtime
COPY requirements.txt .
RUN pip install --requirement requirements.txt
COPY app ./app
USER ortholens
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS test
COPY requirements.txt requirements-test.txt ./
RUN pip install --requirement requirements-test.txt
COPY app ./app
COPY tests ./tests
USER ortholens
CMD ["pytest", "-q"]

