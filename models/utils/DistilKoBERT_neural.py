import torch.nn as nn
from .model_manager import ModelManager
from tracking.save_registry import SaveTracking

class DistilBERTClassifier(nn.Module):
    def __init__(self, model_module, model_config, model_name, num_labels):
        super(DistilBERTClassifier, self).__init__()
        
        model_manager = ModelManager(model_name)
        self.id2label = model_manager.id2label
        self.label2id = model_manager.label2id
        
        # Pretrained 모델 로드 및 Config 업데이트
        self.model_config = model_config.from_pretrained(model_name, id2label=self.id2label, label2id=self.label2id, num_labels=num_labels)
        self.model_module = model_module.from_pretrained(
            model_name,
            config=self.model_config,
            trust_remote_code=True
        )
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.model_module.config.hidden_size, num_labels)
        
        # 손실 함수는 한 번만 정의
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask, labels=None):
        # DistilBERT Forward Pass
        outputs = self.model_module(input_ids=input_ids, attention_mask=attention_mask)
        
        # [CLS] 토큰 추출
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        # 분류기 통과
        logits = self.classifier(cls_output)
        
        # 손실 계산 (Train 단계)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        return {
            "loss": loss,
            "logits": logits
        }
