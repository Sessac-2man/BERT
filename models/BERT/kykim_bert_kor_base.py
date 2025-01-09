from datasets import Dataset, load_dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import mlflow
import mlflow.sklearn
import numpy as np

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2, ignore_mismatched_sizes=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.learning_rate = 2e-5
        self.epochs = 3
        self.train_batch_size = 16
        self.valid_batch_size = 16
        self.experiment = 'bert_kor_base'

        mlflow.set_tracking_uri('http://localhost:5001')
        mlflow.set_experiment(self.experiment)

    def train(self, train_tokenized_dataset, val_tokenized_dataset):
        self.training_args = TrainingArguments(
            output_dir='./result',
            evaluation_strategy='epoch',
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.valid_batch_size,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
            # logging_dir='./logs',
            save_total_limit=2,
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_tokenized_dataset,
            eval_dataset=val_tokenized_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        mlflow.start_run()
        mlflow.log_param('learning_rate', self.learning_rate)
        mlflow.log_param('epochs', self.epochs)
        mlflow.log_param('train_batch_size', self.train_batch_size)
        mlflow.log_param('valid_batch_size', self.valid_batch_size)

        print("🚀 Starting Training...")
        self.trainer.train()

        print("📊 Running Validation...")
        eval_results = self.trainer.evaluate()
        print("✅ Validation Results:")
        for key, value in eval_results.items():
            print(f"{key}: {value:.4f}")
            mlflow.log_metric(key, value)

        print("💾 Saving Best Model and Tokenizer...")
        self.trainer.save_model('./result')
        self.tokenizer.save_pretrained('./result')
        print('Model saved at:', self.training_args.output_dir)

        mlflow.pytorch.log_model(self.model, artifact_path=f'{self.experiment}/model')
        mlflow.log_artifacts('./result', artifact_path=f'{self.experiment}/artifacts')

    def test(self, test_dataset):
        self.predictions = []

        for i in range(len(test_dataset)):
            self.text = test_dataset['text'].values[i]
            self.test_encoding = self.tokenizer(self.text, truncation=True, padding=True, max_length=64, return_tensors='pt')
            self.test_data = Dataset.from_dict(self.test_encoding)
            self.pred = self.trainer.predict(self.test_data)
            self.predictions.append(self.pred.predictions.argmax(axis=-1)[0])

        self.score = accuracy_score(test_dataset['label'], self.predictions)
        print(self.score)

        return self.score
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)

        accuracy = accuracy_score(labels, predictions)

        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

        return {
            'accuracy' : accuracy,
            'precision' : precision,
            'recall' : recall,
            'f1' : f1
        }

    def get_data(self):
        self.dataset = load_dataset('UICHEOL-HWANG/hate_speech')
        # Train Dataset
        self.train_df = pd.DataFrame(self.dataset['train'])
        self.train_df = self.train_df
        self.train_tokenized_dataset = Dataset.from_pandas(self.train_df).map(self.preprocess_function, batched=True)

        # Validation Dataset
        self.val_dataset = pd.read_csv('C:/Users/user/Desktop/BERT/result/test.tsv', sep='\t') # Set Your Validation Dataset Route
        self.val_dataset = self.val_dataset[:5000]
        self.val_dataset['label'] = self.val_dataset['label'].apply(lambda x:0 if x == 1 else 1)
        self.val_tokenized_dataset = Dataset.from_pandas(self.val_dataset).map(self.preprocess_function, batched=True)

        # Test Dataset
        self.test_df = pd.DataFrame(self.dataset['test'])

        return self.train_tokenized_dataset, self.val_tokenized_dataset, self.test_df

    def preprocess_function(self, examples):
        return self.tokenizer(examples['text'], truncation=True, padding=True, max_length=128)

if __name__ == '__main__':
    model = Model('kykim/bert-kor-base')
    train, val, test = model.get_data()
    model.train(train, val)
    score = model.test(test)
    print(score)