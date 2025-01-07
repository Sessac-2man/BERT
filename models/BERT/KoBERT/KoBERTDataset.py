from torch.utils.data import Dataset 
import torch

class KoBERTDataset(Dataset):
    
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data 
        self.tokenizer= tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data['text'][idx]
        label = self.data['label'][idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            "input_ids" : encoding['input_ids'].squeeze(0),
            "attention_mask" : encoding['attention_mask'].squeeze(0),
            "labels" : torch.tensor(label, dtype=torch.long)
        }