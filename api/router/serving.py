
from .serving_schemas import TextInput, ClassificationResult
from inference.inference import load_model
from utils.connect import Connect

import os 
from fastapi import APIRouter, HTTPException
import logging
from typing import List
import mlflow

tracking = Connect()

# 환경 변수 설정
os.environ["MLFLOW_S3_ENDPOINT_URL"] = tracking.mlflow_s3_endpoint
os.environ["MLFLOW_TRACKING_URI"] = tracking.mlflow_tracking_uri
os.environ["AWS_ACCESS_KEY_ID"] =  tracking.minio_access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = tracking.minio_secret_key

# MLflow 런 ID와 아티팩트 경로 설정

run_id = tracking.serach_best_run() # 실제 Run ID로 변경
artifact_path = "outputs"  # 모델 저장 시 사용한 artifact_path와 일치

# MLflow 아티팩트 URI 구성
model_uri = f"runs:/{run_id}/{artifact_path}"

router = APIRouter(
    prefix="/api/inference",
    tags=["Inference"]
)   


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.on_event("startup")
def startup_event():
    global classifier
    try:
        classifier = mlflow.transformers.load_model(model_uri=model_uri)
        logger.info("✅ 모델이 성공적으로 로드되었습니다.")
    except mlflow.exceptions.MlflowException as e:
        logger.error(f"❌ 모델 로드 중 오류 발생: {e}")
        raise e 
  
@router.post("/classify", response_model=List[ClassificationResult])
def classify_text(input: TextInput):
    try:
        # 분류 수행
        results = classifier(input.texts)
        
        response = [] 
        
        for text, res in zip(input.texts, results):
            # 디버깅을 위한 로그 추가
            logger.info(f"Input Text: {text}")
            logger.info(f"Model Label: {res['label']}")
            logger.info(f"Model Score: {res['score']}")
            
            # 레이블 매핑 없이 직접 할당
            label = str(res['label'])  # 레이블이 정수일 경우 문자열로 변환
            score = res['score']
            response.append(ClassificationResult(text=text, label=label, score=score))
            
        return response

    except Exception as e:
        logger.error(f"❌ 분류 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
