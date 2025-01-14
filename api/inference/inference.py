from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR = "../results/"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# 모델을 평가 모드로 전환
model.eval()

# 샘플 입력
sample_text = "안녕하세요"

# 토크나이저로 입력 텍스트 토큰화
inputs = tokenizer(
    sample_text,
    return_tensors="pt",  # PyTorch 텐서로 반환
    padding=True,         # 패딩 추가
    truncation=True,      # 문장 자르기
    max_length=512        # 최대 길이 제한
)

# 모델에 입력 데이터 전달
with torch.no_grad():  # 그래디언트를 계산하지 않음 (평가 모드)
    outputs = model(**inputs)
    logits = outputs.logits  # 분류 결과 logits

# 결과 확인
predicted_class = torch.argmax(logits, dim=1).item()  # 가장 높은 확률의 클래스
print(f"🔍 입력 문장: {sample_text}")
print(f"✅ 예측된 클래스: {predicted_class}")