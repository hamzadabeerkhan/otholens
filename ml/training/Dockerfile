FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN addgroup --system ortholens && adduser --system --ingroup ortholens ortholens
COPY ml/training/requirements.txt requirements.txt
RUN pip install --requirement requirements.txt
COPY ml/training ./training
USER ortholens
ENTRYPOINT ["python", "-m", "training.train"]
