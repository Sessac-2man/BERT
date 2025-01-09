import torch
from transformers import (
    Trainer,
    TrainingArguments
)
from sklearn.metrics import f1_score, precision_score, recall_score
import mlflow, mlflow.pytorch
from tracking.save_registry import SaveTracking


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
        mlflow.set_tracking_uri(SaveTracking().mlflow_url)
        mlflow.set_experiment(self.experiment)
        print(f"실험 설정 완료 실험버전명 {self.experiment}")
    
    def train(self, train_dataset, valid_dataset, output_dir, train_batch_size, valid_batch_size):
        """
        모델 학습, 검증 및 최적 모델 저장
        """
        training_args = TrainingArguments(
            output_dir=output_dir,             # 결과 저장 경로
            num_train_epochs=self.epochs,       # 학습 Epoch 수
            per_device_train_batch_size=train_batch_size,     # GPU당 학습 배치 크기
            per_device_eval_batch_size=valid_batch_size,      # GPU당 검증 배치 크기
            warmup_steps=100,                   # 학습률 스케줄링을 위한 웜업 스텝
            weight_decay=0.01,                  # 가중치 감소
            logging_dir='./logs',               # 로그 저장 경로
            logging_steps=10,                   # 로그 출력 간격
            evaluation_strategy='steps',        # 매 Epoch마다 Validation 실행
            eval_steps=500,
            save_strategy='steps',              # 매 Epoch마다 체크포인트 저장
            load_best_model_at_end=True,        # 최적의 모델 불러오기
            metric_for_best_model='f1',         # 최적 모델 기준
            fp16=True if torch.cuda.is_available() else False,  # Mixed Precision 활성화
            report_to='none',                   # TensorBoard 비활성화
            logging_first_step=True,             # 첫 스텝부터 로그 출력
            max_grad_norm=1.0   
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
            # 학습 실행
            print("🚀 Starting Training...")
            trainer.train()
        
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
            
            mlflow.pytorch.log_model(self.model, artifact_path=f"{self.experiment}/model")
            mlflow.log_artifacts(output_dir, artifact_path=f"{self.experiment}/artifacts")
            
            print(f"모델 저장완로: {self.experiment}/model")
            print(f"아티팩트 저장완료: {self.experiment}/artifacts")
            
        return eval_results
