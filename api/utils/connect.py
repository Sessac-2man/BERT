import psycopg2
from dotenv import load_dotenv
import os 

class Connect:
    def __init__(self):
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # dotenv_path = os.path.join(current_dir, "C:/Users/user/Desktop/BERThub/Labs/.env")
        # load_dotenv(dotenv_path=dotenv_path)
        
        # host connect 
        
        self.host = os.getenv("POSTGRES_HOST")
        self.port = os.getenv("POSTGRES_PORT")
        self.dbname = os.getenv("POSTGRES_DB")
        self.user = os.getenv("POSTGRES_USER")
        self.password = os.getenv("POSTGRES_PASSWORD")
       # MLflow 설정 정보
       
        self.mlflow_tracking_uri = os.getenv("MLFLOW_URL")  # MLflow Tracking Server URI
        self.mlflow_s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT")  # MinIO S3 엔드포인트

        # MinIO 설정 정보
        self.minio_access_key = os.getenv("AWS_ACCESS_KEY_ID")  # MinIO Access Key
        self.minio_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")  # MinIO Secret Key
        
    def serach_best_run(self):
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.dbname,
                user=self.user,
                password=self.password
            )
            print("✅ Database connection successful")
            cursor = conn.cursor()
            
            query = """
            SELECT run_uuid
            FROM metrics
            WHERE key = 'loss'
            ORDER BY value ASC
            LIMIT 1;
            """
            cursor.execute(query=query)
            result = cursor.fetchone()
            print(f"Query result: {result}")  # 쿼리 결과 출력
            
            if result:
                best_run_uuid = result[0]
                print(f"Best run UUID: {best_run_uuid}")  # 디버깅용 출력
                return best_run_uuid
            else:
                print("No results found for the query.")
                return None
        except Exception as e:
            print(f"커넥트 실패 {e}")

        
        