import pandas as pd
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np

# Setup stopword & stemmer
stemmer = StemmerFactory().create_stemmer()
stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = stemmer.stem(text)
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

def preprocess_all(texts):
    with Pool(cpu_count()) as p:
        result = list(tqdm(p.imap(clean_text, texts), total=len(texts)))
    return result

# Tambahkan ini agar multiprocessing bisa jalan di Windows
if __name__ == '__main__':
    # Load data
    df = pd.read_excel("./data/dataset_turnbackhoax_10k.xlsx")

    # Jalankan preprocessing
    df['cleaned_text'] = preprocess_all(df['text'].astype(str))

    # Simpan hasil
    df.to_excel("dataset_turnbackhoax_10k_cleaned.xlsx", index=False)
    print("Preprocessing selesai dan disimpan.")
