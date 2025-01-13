import psycopg2 
from dotenv import load_dotenv
import os 

class Connect:
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
        
    def serach_best_run(self):
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.dbname,
                user=self.user
            )
            cursor = conn.cursor()
            
            query = """
            SELECT run_uuid
            FROM metrics
            WHERE key = 'loss'
            ORDER BY value ASC
            LIMIT 1; 
            """
            
            cursor.execute(query=query)
            best_run_uuid = cursor.fetchone()[0]
            
            return best_run_uuid
        except Exception as e:
            print(f"Error : {e}")
            return None
        
        