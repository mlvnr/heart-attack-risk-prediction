# 🫀 Heart Attack Risk Prediction — Streamlit App

Aplikasi web untuk memprediksi risiko serangan jantung menggunakan model machine learning.
**Group 5 (Pandas) · Data Science Batch 60 · Digital Skola**

---

## 📁 Struktur Folder

Pastikan struktur folder seperti ini sebelum menjalankan:

```
streamlit_app/
├── app.py                      # aplikasi utama
├── requirements.txt            # dependencies
├── README.md                   # file ini
└── artifacts/
    ├── best_pipeline.pkl       # pipeline (preprocessor + classifier)
    └── metadata.pkl            # info model + daftar kolom input
```

> **Penting:** Folder `artifacts/` dihasilkan dari notebook (Section 5).
> Jalankan notebook sampai selesai, download `artifacts.zip`, ekstrak di sini.

---

## 🚀 Cara Menjalankan

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Pastikan folder artifacts ada

Copy `best_pipeline.pkl` dan `metadata.pkl` dari hasil notebook ke folder `artifacts/`.

### 3. Jalankan aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser (biasanya `http://localhost:8501`).

---

## 🖥️ Cara Pakai

1. Isi form data pasien (10 field):
   - **Pemeriksaan klinis:** Kolesterol, Tekanan Sistolik/Diastolik, Trigliserida
   - **Gaya hidup:** Jam Tidur, Jam Olahraga, Pendapatan
   - **Riwayat & kondisi:** Diabetes, Obesitas, Konsumsi Alkohol (Ya/Tidak)
2. Klik tombol **"🔍 Prediksi Risiko"**
3. Lihat hasil: **Risk** / **No Risk** beserta probabilitasnya

---

## 🌐 Deploy ke Streamlit Cloud (opsional)

1. Push folder ini ke GitHub (termasuk `artifacts/`)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect repo, pilih `app.py` sebagai main file
4. Deploy — aplikasi jadi bisa diakses publik via URL

---

## ⚠️ Disclaimer

Model dilatih pada **dataset synthetic** (bukan data pasien nyata) dan hanya untuk
tujuan pembelajaran. Akurasi terbatas (~50-65%). Hasil prediksi **TIDAK boleh**
dijadikan dasar diagnosis medis.

---

## 🔧 Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `artifacts/best_pipeline.pkl tidak ditemukan` | Jalankan notebook Section 5, ekstrak `artifacts.zip` ke folder ini |
| `ModuleNotFoundError` | Jalankan `pip install -r requirements.txt` |
| Prediksi error | Pastikan versi scikit-learn/imblearn sama dengan yang dipakai saat training |
| Model versi mismatch | Re-generate artifacts dengan versi library yang sama dengan requirements.txt |
