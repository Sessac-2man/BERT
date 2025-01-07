from transformers import BertForSequenceClassification, BertConfig
from tokenizer.KoBertTokenizer import KoBertTokenizer




class ModelManager:
    def __init__(self):
        # 라벨 설정
        self.id2label = {0: "혐오", 1: "일상"}
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        # 모델 경로 및 설정
        self.model_path = "monologg/kobert"        
        
    def initialized_model(self, num_labels):
        config = BertConfig.from_pretrained(self.model_path, id2label=self.id2label, label2id=self.label2id, num_labels=num_labels)
        model =  BertForSequenceClassification.from_pretrained(
            self.model_path,
            config=config,
            trust_remote_code=True
        )
        
        tokenizer =  KoBertTokenizer.from_pretrained(self.model_path)
        
        return model, tokenizer
        

    