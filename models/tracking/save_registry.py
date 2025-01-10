import psycopg2 
import os 
from dotenv import load_dotenv
import sys 



class SaveTracking:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dotenv_path = os.path.join(current_dir, "C:/Users/user/Desktop/BERThub/Labs/.env")
        load_dotenv(dotenv_path=dotenv_path)
        
        # host connect 
        
        self.host = os.getenv("POSTGRES_HOST")
        self.port = os.getenv("POSTGRES_PORT")
        self.dbname = os.getenv("POSTGRES_DB")
        self.user = os.getenv("POSTGRES_USER")
        self.password = os.getenv("POSTGRES_PASSWORD")
        
        self.mlflow_url = os.getenv("MLFLOW_URL")
        self.bucket_url = os.getenv("BUCKET_URL")
        self.minio = os.getenv("MINIO_ROOT_USER")
        self.minio_key = os.getenv("MINIO_ROOT_PASSWORD")
        self.s3 = os.getenv("MLFLOW_ARTIFACT_STORE_URI")
    def connect_to_database(self):
        try:
            connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.user,
                password=self.password
            )
            print("데이터 베이스 연결완료")
            return connection
        except Exception as e:
            print(f"데이터베이스 연결 오류 {e}")
            sys.exit(1)
    
        