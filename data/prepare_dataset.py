import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from datasets import Dataset, DatasetDict

# 1. Baca dataset bersih dan terlabel
df = pd.read_excel("dataset_cleaned_labeled.xlsx")

# 2. Hapus baris yang kosong di kolom penting
df = df.dropna(subset=['cleaned_text', 'label'])

# 3. Pastikan tipe data kolom teks adalah string
df['cleaned_text'] = df['cleaned_text'].astype(str)

# 4. Split data ke training dan testing (80:20)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['cleaned_text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# 5. Load tokenizer IndoBERT
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

# 6. Fungsi tokenisasi
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# 7. Buat HuggingFace Dataset dari dictionary
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

# 8. Gabungkan menjadi DatasetDict
dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# 9. Tokenisasi dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 10. Simpan hasil tokenisasi ke disk
tokenized_dataset.save_to_disk("tokenized_dataset_indobert")

print("Dataset berhasil diproses dan disimpan di 'tokenized_dataset_indobert'")
