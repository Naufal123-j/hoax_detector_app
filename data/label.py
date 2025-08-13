import pandas as pd
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# 1. Load dataset
df = pd.read_excel("dataset_turnbackhoax_10k_cleaned.xlsx")

# 2. Buat label otomatis berdasarkan kolom 'Tags'
def label_from_tags(tag):
    if pd.isna(tag):
        return None
    tag_list = [t.strip().lower() for t in tag.split(';')]
    if any(keyword in tag_list for keyword in ['hoax', 'fitnah', 'hasut']):
        return 1  # hoaks
    elif any(keyword in tag_list for keyword in ['fakta', 'klarifikasi', 'benar', 'valid']):
        return 0  # fakta
    else:
        return None

df['label'] = df['Tags'].apply(label_from_tags)

# 3. Hapus data yang tidak punya label
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# 4. Preprocessing teks
stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)                     # hapus url
    text = re.sub(r'@\w+', '', text)                        # hapus mention
    text = re.sub(r'#\w+', '', text)                        # hapus hashtag
    text = re.sub(r'\d+', '', text)                         # hapus angka
    text = text.translate(str.maketrans('', '', string.punctuation))  # hapus tanda baca
    text = stopword.remove(text)
    text = stemmer.stem(text)
    return text

# Terapkan pembersihan pada kolom 'text' atau 'cleaned_text' awal
if 'cleaned_text' not in df.columns:
    df['cleaned_text'] = df['text'].apply(clean_text)

# 5. Simpan dataset siap pakai
df[['cleaned_text', 'label']].to_excel("dataset_cleaned_labeled.xlsx", index=False)

print("Dataset berhasil diproses dan disimpan sebagai 'dataset_cleaned_labeled.xlsx'")
print(f"Total data terpakai: {len(df)} baris")
