FROM python:3.11.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir --upgrade -r requirements_detailed.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]