import mlflow
from transformers import pipeline
import os 



def load_model(run_id: str, artifact_path: str, model_uri):
    try:
        loaded_pipeline = mlflow.transformers.load_model(model_uri=model_uri)
        print("모델이 성공적으로 로드되었습니다.")
        return loaded_pipeline
    except mlflow.exceptions.MlflowException as e:
        print(f"모델 로드 중 오류: {e}")
        raise e 
    

