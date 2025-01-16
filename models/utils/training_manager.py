import torch
from transformers import (
    Trainer,
    TrainingArguments,
    pipeline
)
from sklearn.metrics import f1_score, precision_score, recall_score
import mlflow
from tracking.save_registry import SaveTracking
import os 

tracking = SaveTracking()

os.environ["MLFLOW_S3_ENDPOINT_URL"] = tracking.mlflow_s3_endpoint
os.environ["MLFLOW_TRACKING_URI"] = tracking.mlflow_tracking_uri
os.environ["AWS_ACCESS_KEY_ID"] =  tracking.minio_access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = tracking.minio_secret_key

# GPU 확인 함수
def check_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No GPU Detected. Using CPU instead.")
    return device


# 평가 지표 함수
def compute_metrics(eval_pred):
    """
    검증 시 F1 Score, Precision, Recall을 계산합니다.
    """
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    labels = labels
    
    f1 = f1_score(labels, preds, average='macro')
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# TrainingManager 클래스
class TrainingManager:
    def __init__(self, model, tokenizer, learning_rate, experiment, epochs=5):
        self.learning_rate = learning_rate
        self.model = model
        self.tokenizer = tokenizer
        self.device = check_device()
        self.model.to(self.device)
        self.epochs = epochs
        self.experiment = experiment
        
        # 실험 세팅
        mlflow.set_tracking_uri(tracking.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment)
        print(f"실험 설정 완료 실험버전명 {self.experiment}")
    
    def train(self, train_dataset, 
              valid_dataset, 
              output_dir, 
              train_batch_size, 
              valid_batch_size,
              weight_decay,
              grad_norm,
              logging_dir
              ):
        """
        모델 학습, 검증 및 최적 모델 저장
        """
        training_args = TrainingArguments(
            output_dir=output_dir,             # 결과 저장 경로
            num_train_epochs=self.epochs,       # 학습 Epoch 수
            per_device_train_batch_size=train_batch_size,     # GPU당 학습 배치 크기
            per_device_eval_batch_size=valid_batch_size,      # GPU당 검증 배치 크기
            warmup_steps=100,                   # 학습률 스케줄링을 위한 웜업 스텝
            weight_decay=weight_decay,                  # 가중치 감소
            logging_dir=logging_dir,               # 로그 저장 경로
            logging_steps=10,                   # 로그 출력 간격
            evaluation_strategy='steps',        # 매 Epoch마다 Validation 실행
            eval_steps=500,
            save_strategy='steps',              # 매 Epoch마다 체크포인트 저장
            load_best_model_at_end=True,        # 최적의 모델 불러오기
            metric_for_best_model='f1',         # 최적 모델 기준
            fp16=True if torch.cuda.is_available() else False,  # Mixed Precision 활성화
            report_to='mlflow',                   # mlflow로 세팅
            logging_first_step=True,             # 첫 스텝부터 로그 출력
            max_grad_norm=grad_norm 
        )
        
        # Trainer 객체 생성
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )
        
        with mlflow.start_run():
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("train_batch_size", train_batch_size)
            mlflow.log_param("valid_batch_size", valid_batch_size)
            mlflow.log_param("max_grad_norm", grad_norm)

            
            
            # 학습 실행
            print("🚀 Starting Training...")
            training = trainer.train()

            mlflow.log_metric("train_loss", training.metrics["train_loss"])
            mlflow.log_metric("train_runtime", training.metrics["train_runtime"])
            mlflow.log_metric("train_samples_per_second", training.metrics["train_samples_per_second"])
        
            # 검증 실행
            print("📊 Running Validation...")
            eval_results = trainer.evaluate()
            print("✅ Validation Results:")
            for key, value in eval_results.items():
                print(f"{key}: {value:.4f}")
                # 로깅 결과 수집 
                mlflow.log_metric(key, value)
        
            # 최적 모델 및 토크나이저 저장
            print("💾 Saving Best Model and Tokenizer...")
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"✅ Model and Tokenizer Saved at {output_dir}")
            
            task = "text-classification"
            
            sentence_pipeline = pipeline(
                task=task,
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1 
            )
            
            mlflow.transformers.log_model(
                transformers_model=sentence_pipeline,
                artifact_path="outputs",
                task=task,
                registered_model_name=self.experiment
            )
            
            print(f"아티팩트 저장완료: {self.experiment}/outputs")
            
        return eval_results
