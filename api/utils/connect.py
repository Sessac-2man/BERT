import psycopg2
import os 

class Connect:
    def __init__(self):

        # 데이터베이스 연결용 클래스
        
        self.host = os.getenv("POSTGRES_HOST")
        self.port = os.getenv("POSTGRES_PORT")
        self.dbname = os.getenv("POSTGRES_DB")
        self.user = os.getenv("POSTGRES_USER")
        self.password = os.getenv("POSTGRES_PASSWORD")

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

        
        