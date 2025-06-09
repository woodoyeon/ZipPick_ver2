# ▶ Python 이미지 불러오기
FROM python:3.10

# ▶ 작업 디렉토리 설정
WORKDIR /app

# ▶ 필요한 파일 복사
COPY requirements.txt .
COPY app_pro.py .
COPY .env .

# ▶ 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# ▶ 포트 열기 (Streamlit 기본 포트)
EXPOSE 8501

# ▶ 앱 실행
CMD ["streamlit", "run", "app_pro.py", "--server.port=8501", "--server.address=0.0.0.0"]
