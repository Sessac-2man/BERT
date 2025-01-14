import psycopg2
import os
from dotenv import load_dotenv
import sys


class SaveTracking:
    def __init__(self):
        # .env 파일 로드
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dotenv_path = os.path.join(current_dir, "C:/Users/user/Desktop/BERThub/Labs/.env")
        load_dotenv(dotenv_path=dotenv_path)

        # 데이터베이스 연결 정보
        self.db_host = os.getenv("POSTGRES_HOST")  # PostgreSQL 호스트
        self.db_port = os.getenv("POSTGRES_PORT")  # PostgreSQL 포트
        self.db_name = os.getenv("POSTGRES_DB")    # PostgreSQL 데이터베이스 이름
        self.db_user = os.getenv("POSTGRES_USER")  # PostgreSQL 사용자 이름
        self.db_password = os.getenv("POSTGRES_PASSWORD")  # PostgreSQL 비밀번호

        # MLflow 설정 정보
        self.mlflow_tracking_uri = os.getenv("MLFLOW_URL")  # MLflow Tracking Server URI
        self.mlflow_s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT")  # MinIO S3 엔드포인트

        # MinIO 설정 정보
        self.minio_access_key = os.getenv("AWS_ACCESS_KEY_ID")  # MinIO Access Key
        self.minio_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")  # MinIO Secret Key

    def connect_to_database(self):
        """
        PostgreSQL 데이터베이스에 연결
        """
        try:
            connection = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            print("✅ 데이터베이스 연결 완료")
            return connection
        except Exception as e:
            print(f"❌ 데이터베이스 연결 오류: {e}")
            sys.exit(1)
