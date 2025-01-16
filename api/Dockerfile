FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# pip 업데이트
RUN pip install --upgrade pip

# PyTorch 설치 (CPU 버전)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 의존성 설치
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 복사
COPY . .

CMD ["python", "main.py"]
