from utils.model_manager import ModelManager
from utils.training_manager import TrainingManager
from utils.KoBERTDataset import KoBERTDataset
from utils.DistilKoBERT_neural import DistilBERTClassifier

from transformers import BertModel, DistilBertModel
import argparse
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Train KcBERT model")
    
    # Argument 설정
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--train_data', type=str, default='train', help='Training dataset split')
    parser.add_argument('--valid_data', type=str, default='test', help='Validation dataset split')
    parser.add_argument('--output_dir', type=str, default='./KcOutput', help='Directory to save the model')
    args = parser.parse_args()
    
    tokenizer = DistilBertModel.from_pretrained('monologg/distilkobert')
    
    # ✅ 모델 설정
    model = DistilBERTClassifier(
        model_module=BertModel,
        model_name='monologg/distilkobert',
        num_labels=2  # 예: 긍정/부정 분류
    )
    
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
    results = training_manager.train(train_dataset, valid_dataset, args.output_dir)
    
    print("✅ Final Validation Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")


if __name__ == '__main__':
    main()
