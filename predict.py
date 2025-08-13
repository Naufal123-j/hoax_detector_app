from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd

# Load tokenizer & model dari folder ./model
model_path = "./model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Pastikan model ke eval mode
model.eval()

# Fungsi prediksi
def predict_hoax(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        probs = torch.nn.functional.softmax(logits, dim=-1)
    return predictions.tolist(), probs.tolist()

# Contoh input
sample_texts = [
    "Pemerintah membagikan uang tunai Rp50 juta ke seluruh warga desa",
    "Presiden menghadiri upacara hari kemerdekaan di IKN"
]

# Prediksi
labels, probabilities = predict_hoax(sample_texts)

# Tampilkan hasil
for i, text in enumerate(sample_texts):
    label = "Hoaks" if labels[i] == 1 else "Fakta"
    confidence = max(probabilities[i]) * 100
    print(f"Teks: {text}")
    print(f"Prediksi: {label} (Confidence: {confidence:.2f}%)\n")
