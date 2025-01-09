from utils.model_manager import ModelManager
from utils.training_manager import TrainingManager
from utils.KoBERTDataset import KoBERTDataset
from utils.DistilKoBERT_neural import DistilBERTClassifier

from transformers import DistilBertModel, DistilBertTokenizer
import argparse
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Train DistilBERT model")
    
    # Argument 설정
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--train_data', type=str, default='train', help='Training dataset split')
    parser.add_argument('--valid_data', type=str, default='test', help='Validation dataset split')
    parser.add_argument('--output_dir', type=str, default='./KcOutput', help='Directory to save the model')
    parser.add_argument('--experiment', type=str, default='DistilBERT_v1', help='experiment set name')
    args = parser.parse_args()
    
    tokenizer = DistilBertTokenizer.from_pretrained('monologg/distilkobert')
    
    # ✅ 모델 설정
    model = DistilBERTClassifier(
        model_module=DistilBertModel,
        model_name='monologg/distilkobert',
        num_labels=2  # 예: 긍정/부정 분류
    )
    
    training_manager = TrainingManager(
        model=model, 
        tokenizer=tokenizer, 
        learning_rate=args.learning_rate, 
        experiment=args.experiment,
        epochs=args.epochs
    )
    
    # ✅ 데이터셋 로드 및 토큰화
    dataset = load_dataset('Sessac-Blue/hate-speech')
    train_dataset = KoBERTDataset(dataset[args.train_data], tokenizer=tokenizer, use_token_type_ids=False, max_length=128)
    valid_dataset = KoBERTDataset(dataset[args.valid_data], tokenizer=tokenizer, use_token_type_ids=False, max_length=128)
    # 🚀 Training & Validation
    
    print("🚀 Starting Training and Validation...")
    print(f"Training DistilBERT model with the following configuration:")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Train Split: {args.train_data}")
    print(f"  Validation Split: {args.valid_data}")
    print(f"  Output Directory: {args.output_dir}")
    
    results = training_manager.train(train_dataset, valid_dataset, output_dir=args.output_dir, train_batch_size=args.batch_size, valid_batch_size=args.batch_size)
    
    print("✅ Final Validation Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")


if __name__ == '__main__':
    main()
