import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="VerifAI", layout="centered")
st.title("🔎 Deteksi Berita Hoaks dengan IndoBERT")
st.markdown("Masukkan teks berita yang ingin Anda verifikasi:")

text_input = st.text_area("Teks berita", height=200)

if st.button("Prediksi"):
    if text_input:
        with st.spinner("Menganalisis..."):
            try:
                res = requests.post("http://localhost:5000/predict", json={"text": text_input})
                if res.status_code == 200:
                    hasil = res.json()
                    st.success(f"Prediksi: **{hasil['prediction']}** (Keyakinan: {hasil['confidence']*100:.2f}%)")
                else:
                    st.error("Gagal memproses permintaan ke server Flask.")
            except Exception as e:
                st.error(f"Error koneksi: {e}")
    else:
        st.warning("Mohon masukkan teks terlebih dahulu.")

# 🔹 Form upload file
uploaded_file = st.file_uploader("dataset_turnbackhoax_10k.xlsx", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)

        if 'text' not in df.columns:
            st.error("File harus memiliki kolom 'text'")
        else:
            st.write("📄 Data berhasil dimuat:")
            st.dataframe(df.head())

            # 🔹 Kirim setiap baris ke API Flask
            results = []
            for idx, row in df.iterrows():
                text = row['text']
                response = requests.post("http://127.0.0.1:5000/predict", json={"text": text})
                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        "text": text,
                        "prediction": result["prediction"],
                        "confidence": result["confidence"]
                    })

            result_df = pd.DataFrame(results)
            st.success("✅ Hasil Prediksi:")
            st.dataframe(result_df)

            # Unduh hasil sebagai Excel
            hasil_excel = result_df.to_excel(index=False, engine='openpyxl')
            st.download_button("⬇️ Unduh Hasil ke Excel", data=hasil_excel, file_name="hasil_prediksi.xlsx")

    except Exception as e:
        st.error(f"Gagal memproses file: {e}")