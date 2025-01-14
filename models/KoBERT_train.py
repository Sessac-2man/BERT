from utils.model_manager import ModelManager
from utils.training_manager import TrainingManager
from utils.KoBERTDataset import KoBERTDataset
from tokenizer.KoBertTokenizer import KoBertTokenizer

from transformers import BertForSequenceClassification, BertConfig
import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Train KoBERT model")
    
    # Argument 설정
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--train_data', type=str, default='train', help='Training dataset split')
    parser.add_argument('--valid_data', type=str, default='test', help='Validation dataset split')
    parser.add_argument('--output_dir', type=str, default='./KoOutput', help='Directory to save the model')
    parser.add_argument('--experiment', type=str, default='KoBERT_v1', help='experiment set name')
    parser.add_argument('--weight_decay', type=float, default='0.01', help='영어 쓰기 귀찮아서 한글 쓴다 가중치 제한(감소)')
    parser.add_argument('--grad_norm', type=float, default='1.0', help='그래디언트 클리핑 제한 ')
    parser.add_argument('--logging_dir', type=str, default='./logs', help='saving log')
    args = parser.parse_args()
    
    # ✅ Model Manager 초기화
    model_manager = ModelManager("monologg/kobert")
    model, tokenizer = model_manager.initialized_model(model_config=BertConfig, model_pretrained=BertForSequenceClassification, tokenize=KoBertTokenizer, num_labels=2)
    training_manager = TrainingManager(
        model=model, 
        tokenizer=tokenizer, 
        learning_rate=args.learning_rate, 
        epochs=args.epochs
    )
    
    # ✅ 데이터셋 로드 및 토큰화
    dataset = load_dataset('Sessac-Blue/hate-speech')
    train_dataset = KoBERTDataset(dataset[args.train_data], tokenizer=tokenizer, use_token_type_ids=True, max_length=128)
    valid_dataset = KoBERTDataset(dataset[args.valid_data], tokenizer=tokenizer, use_token_type_ids=True, max_length=128)
    
    # 🚀 Training & Validation
    print("🚀 Starting Training and Validation...")
    
    results = training_manager.train(train_dataset, 
                                valid_dataset, 
                                output_dir=args.output_dir, 
                                train_batch_size=args.batch_size, 
                                valid_batch_size=args.batch_size,
                                weight_decay=args.weight_decay,
                                grad_norm=args.grad_norm
                                )
        
    print("✅ Final Validation Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")


if __name__ == '__main__':
    main()
