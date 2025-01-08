class ModelManager:
    def __init__(self, model_path):
        # 라벨 설정
        self.id2label = {0: "혐오", 1: "일상"}
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        # 모델 경로 및 설정
        self.model_path = model_path    
        
    def initialized_model(self, model_config, model_pretrained,tokenize, num_labels):
        config = model_config.from_pretrained(self.model_path, id2label=self.id2label, label2id=self.label2id, num_labels=num_labels)
        model =  model_pretrained.from_pretrained(
            self.model_path,
            config=config,
            trust_remote_code=True
        )
        
        tokenizer = tokenize.from_pretrained(self.model_path)
        
        return model, tokenizer
        

    