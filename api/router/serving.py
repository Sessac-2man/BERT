
from .serving_schemas import TextInput, ClassificationResult
from inference.inference import load_model
from utils.connect import Connect

import os 
from fastapi import APIRouter, HTTPException
import logging
from typing import List
import mlflow
import pandas as pd


os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_URL")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

db_connect = Connect()

run_id = db_connect.serach_best_run() # 실제 Run ID로 변경
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
        # device 옵션 제거
        classifier = mlflow.pyfunc.load_model(model_uri=model_uri)
        logger.info("✅ 모델이 성공적으로 로드되었습니다.")
    except mlflow.exceptions.MlflowException as e:
        logger.error(f"❌ 모델 로드 중 오류 발생: {e}")
        raise e

  
@router.post("/classify", response_model=List[ClassificationResult])
def classify_text(input: TextInput):
    try:
        logger.info(f"📩 Received input: {input.texts}")

        # pandas DataFrame으로 변환
        input_data = pd.DataFrame({"text": input.texts})

        # 예측 수행
        results = classifier.predict(input_data)  # DataFrame 반환
        logger.info(f"✅ Raw Model Output:\n{results}")

        response = []
        for _, row in results.iterrows():  # DataFrame의 각 행 처리
            label = row["label"]
            score = row["score"] if row["score"] is not None else 0.0  # None 방지

            logger.info(f"📝 Input Text: {input.texts}")
            logger.info(f"🔖 Predicted Label: {label}")
            logger.info(f"📊 Predicted Score: {score}")

            # 응답 생성
            response.append(ClassificationResult(text=input.texts[0], label=label, score=score))

        logger.info("✅ Response successfully generated.")
        return response

    except Exception as e:
        logger.error(f"❌ Error during classification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
