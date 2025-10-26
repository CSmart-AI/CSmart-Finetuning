# Python 3.10 기반 이미지 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 환경변수 설정 (캐시 권한 문제 해결)
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV HF_DATASETS_CACHE=/tmp/hf_cache
ENV HF_HUB_CACHE=/tmp/hf_cache

# 시스템 패키지 업데이트 및 필요한 도구 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 캐시 디렉토리 생성 및 권한 설정
RUN mkdir -p /tmp/hf_cache && chmod 777 /tmp/hf_cache

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY app.py .
COPY gemma2-finetuned ./gemma2-finetuned

# 포트 노출
EXPOSE 7860

# 서버 실행
CMD ["python", "app.py"]

