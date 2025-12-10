# File: app.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- 1. Konfigurasi Halaman Web ---
st.set_page_config(
    page_title="Rekomendasi Produk Kelompok 8",
    layout="wide"
)

st.title('⭐ Sistem Rekomendasi Produk Berbasis Ulasan ⭐')
st.markdown("""
Aplikasi ini merekomendasikan produk berdasarkan kemiripan konten (ulasan) menggunakan metode TF-IDF dan Cosine Similarity.
""")

# --- 2. Pemuatan Data (Ambil dari Colab Anda) ---
# PASTIKAN FILE 'review_data.csv' ADA DI REPOSITORI GITHUB ANDA!
@st.cache_data
def load_data():
    try:
        # Pemuatan data
        df = pd.read_csv("review_data.csv")
        # Pastikan kolom yang digunakan ada
        if 'item_reviewed' not in df.columns or 'text' not in df.columns:
             st.error("Kolom 'item_reviewed' atau 'text' tidak ditemukan dalam data.")
             return None
        return df
    except FileNotFoundError:
        st.error("Gagal memuat review_data.csv. Pastikan file ada di repositori GitHub.")
        return None

df = load_data()

# Lanjutkan hanya jika data berhasil dimuat
if df is not None:

    # --- 3. Pre-processing dan Perhitungan Model ---
    # Logika Model diambil langsung dari Colab Anda.
    # Kita menggunakan semua data karena Streamlit Cloud memiliki memori yang cukup.

    with st.spinner('Menghitung Matriks Kemiripan... Ini mungkin memakan waktu sebentar.'):
        # Mengisi nilai yang hilang (Null) di kolom 'text' dengan string kosong
        df['text_fill'] = df['text'].fillna('')
        
        # Inisialisasi dan Terapkan TF-IDF
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['text_fill'])
        
        # Hitung Cosine Similarity
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # Buat Series untuk pemetaan judul produk ke indeks
        indices = pd.Series(df.index, index=df['item_reviewed']).drop_duplicates()
    
    st.success('Perhitungan model selesai!')


    # --- 4. Fungsi Rekomendasi ---
    def get_recommendations(title, cosine_sim=cosine_sim, df=df, indices=indices):
        # Ambil indeks dari judul produk yang dipilih
        if title not in indices:
            return None
        
        idx = indices[title]
        
        # Dapatkan skor kemiripan dari produk ini dengan semua produk lain
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Urutkan produk berdasarkan skor kemiripan
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Ambil 5 produk paling mirip (indeks 0 adalah produk itu sendiri)
        sim_scores = sim_scores[1:6]
        
        item_indices = [i[0] for i in sim_scores]
        
        # Kembalikan nama-nama produk yang direkomendasikan dan skornya
        results = df[['item_reviewed']].iloc[item_indices]
        results['similarity'] = [i[1] for i in sim_scores]
        return results


    # --- 5. Antarmuka Pengguna (UI) Streamlit ---
    
    product_list = df['item_reviewed'].unique()
    
    # Widget Selectbox
    selected_product = st.selectbox(
        'Pilih produk di bawah ini untuk melihat 5 produk yang paling mirip berdasarkan ulasan:',
        product_list
    )

    # Tombol Aksi
    if st.button('Cari Rekomendasi'):
        if selected_product:
            recommendations = get_recommendations(selected_product)
            if recommendations is not None:
                st.subheader(f'🎯 Top 5 Rekomendasi Mirip dengan **{selected_product}**:')
                # Tampilkan hasil dalam tabel
                st.dataframe(recommendations, use_container_width=True)
            else:
                 st.warning("Produk yang dipilih tidak ditemukan dalam indeks.")

# Tampilkan informasi debug jika diperlukan
# st.sidebar.info(f"Total baris data: {len(df) if df is not None else 0}")
