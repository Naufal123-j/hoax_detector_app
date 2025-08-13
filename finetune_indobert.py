from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd
import torch
import os

# 1. Load dataset
df = pd.read_excel("dataset_cleaned_labeled.xlsx")

# 2. Pastikan kolom 'label' dan 'cleaned_text' tersedia
if 'label' not in df.columns or 'cleaned_text' not in df.columns:
    raise ValueError("Kolom 'label' dan 'cleaned_text' harus ada dalam dataset.")

# 3. Ubah label ke integer
df['label'] = df['label'].astype(int)
df['cleaned_text'] = df['cleaned_text'].astype(str)

# 4. Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['cleaned_text'], df['label'], test_size=0.2, random_state=42
)

# 5. Tokenizer
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

def tokenize_function(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=512)

# 6. Dataset
train_dataset = Dataset.from_dict({'text': train_texts.tolist(), 'label': train_labels.tolist()})
val_dataset = Dataset.from_dict({'text': val_texts.tolist(), 'label': val_labels.tolist()})

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# 7. Load Model
model = BertForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1", num_labels=2)

# 8. Training Arguments
training_args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    dataloader_pin_memory=False  # <-- tambahan ini sangat penting
)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 10. Train Model
print("Mulai training...")
trainer.train()
print("Training selesai.")

# 11. Save Model dan Tokenizer
os.makedirs("./model", exist_ok=True)
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
print("Model berhasil disimpan di ./model")

# 12. (Opsional) Simpan manual bobot model
torch.save(model.state_dict(), "./model/pytorch_model.bin")
print("File pytorch_model.bin berhasil disimpan.")
