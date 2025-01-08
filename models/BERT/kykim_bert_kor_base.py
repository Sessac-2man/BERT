from datasets import Dataset, load_dataset
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2, ignore_mismatched_sizes=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.model.to(self.device)

    def train(self, train_tokenized_dataset, val_tokenized_dataset):
        self.training_args = TrainingArguments(
            output_dir='./result',
            evaluation_strategy='epoch',
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir='./logs',
            save_total_limit=2,
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_tokenized_dataset,
            eval_dataset=val_tokenized_dataset,
            tokenizer=self.tokenizer,
        )

        self.trainer.train()

        self.trainer.save_model()
        self.trainer.save_state()
        print('Model saved at:', self.training_args.output_dir)

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

    def get_data(self):
        self.dataset = load_dataset('UICHEOL-HWANG/hate_speech')
        # Train Dataset
        self.train_df = pd.DataFrame(self.dataset['train'])
        self.train_tokenized_dataset = Dataset.from_pandas(self.train_df).map(self.preprocess_function, batched=True)

        # Validation Dataset
        self.val_dataset = pd.read_csv('Datasets/test.tsv', sep='\t')
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