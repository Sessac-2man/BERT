import torch
from transformers import ElectraForPreTraining, ElectraTokenizer

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
tokenizer.tokenize("[CLS] 한국어 ELECTRA를 공유합니다. [SEP]")