from torch.utils.data import Dataset 
import torch

class KoBERTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128, use_token_type_ids=True):
        """
        Args:
            data (dict): 텍스트와 라벨 데이터
            tokenizer: 토크나이저 객체
            max_length (int): 최대 토큰 길이
            use_token_type_ids (bool): token_type_ids 사용 여부
        """
        self.data = data 
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_token_type_ids = use_token_type_ids  # token_type_ids 사용 여부
    
    def __len__(self):
        return len(self.data['text'])
    
    def __getitem__(self, idx):
        text = self.data['text'][idx]
        label = self.data['label'][idx]
        
        # 동적으로 return_token_type_ids 설정
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_token_type_ids=self.use_token_type_ids,  # 동적 설정
            return_tensors='pt'
        )
        
        # token_type_ids가 있는 경우에만 반환
        item = {
            "input_ids": encoding['input_ids'].squeeze(0),
            "attention_mask": encoding['attention_mask'].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }
        
        if self.use_token_type_ids and 'token_type_ids' in encoding:
            item["token_type_ids"] = encoding['token_type_ids'].squeeze(0)
        
        return item
