"""
Heart Attack Risk Prediction — Streamlit App
Group 5 (Pandas) · Data Science Batch 60

Cara jalankan:
    pip install -r requirements.txt
    streamlit run app.py

Butuh folder artifacts/ berisi:
    - best_pipeline.pkl   (pipeline lengkap: preprocessor + classifier)
    - metadata.pkl        (info model + daftar kolom input)
"""

import streamlit as st
import pandas as pd
import joblib
import os

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Heart Attack Risk Prediction",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# STYLING (tema medis: navy + coral)
# ============================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 800; color: #0F3057;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #6C757D; font-size: 1rem; margin-bottom: 1.5rem; }
    .metric-badge {
        display: inline-block; background: #E1F0F4; color: #00587A;
        padding: 4px 12px; border-radius: 12px; font-size: 0.85rem;
        margin-right: 6px; font-weight: 600;
    }
    .section-header {
        font-size: 1.15rem; font-weight: 700; color: #00587A;
        margin-top: 1rem; margin-bottom: 0.5rem;
        border-bottom: 2px solid #E1F0F4; padding-bottom: 4px;
    }
    .result-risk {
        background: #F8D7DA; border-left: 6px solid #E63946;
        padding: 1.2rem; border-radius: 8px; margin-top: 1rem;
    }
    .result-safe {
        background: #D4EDDA; border-left: 6px solid #28A745;
        padding: 1.2rem; border-radius: 8px; margin-top: 1rem;
    }
    .disclaimer {
        background: #FFF3CD; border-left: 4px solid #FFC107;
        padding: 0.8rem; border-radius: 6px; font-size: 0.85rem;
        color: #664D03; margin-top: 1.5rem;
    }
    .stButton>button {
        background: #E63946; color: white; font-weight: 700;
        border: none; border-radius: 8px; padding: 0.6rem 1rem;
        width: 100%; font-size: 1.05rem;
    }
    .stButton>button:hover { background: #C1121F; color: white; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    """Load pipeline & metadata. Cache supaya tidak reload tiap interaksi."""
    pipeline = joblib.load("artifacts/best_pipeline.pkl")
    metadata = joblib.load("artifacts/metadata.pkl")
    return pipeline, metadata


# ============================================================
# KONFIGURASI WIDGET PER FITUR (label, unit, range, default)
# Metadata-driven: kalau tim ubah TOP_N, app auto-adapt.
# Untuk fitur yang dikenal, pakai range/label yang informatif.
# ============================================================
NUMERIC_CONFIG = {
    "Cholesterol":              {"label": "Kolesterol",            "unit": "mg/dL", "min": 100.0, "max": 450.0, "default": 200.0, "step": 1.0,
                                 "help": "Kadar kolesterol total. Normal < 200, tinggi > 240."},
    "Systolic":                 {"label": "Tekanan Sistolik",      "unit": "mmHg",  "min": 80.0,  "max": 200.0, "default": 120.0, "step": 1.0,
                                 "help": "Angka atas tekanan darah. Normal < 120."},
    "Diastolic":                {"label": "Tekanan Diastolik",     "unit": "mmHg",  "min": 50.0,  "max": 130.0, "default": 80.0,  "step": 1.0,
                                 "help": "Angka bawah tekanan darah. Normal < 80."},
    "Triglycerides":            {"label": "Trigliserida",          "unit": "mg/dL", "min": 20.0,  "max": 800.0, "default": 150.0, "step": 1.0,
                                 "help": "Kadar lemak dalam darah. Normal < 150."},
    "Heart Rate":               {"label": "Detak Jantung",         "unit": "bpm",   "min": 40.0,  "max": 120.0, "default": 72.0,  "step": 1.0,
                                 "help": "Detak jantung istirahat. Normal 60-100 bpm."},
    "BMI":                      {"label": "BMI",                   "unit": "kg/m²", "min": 15.0,  "max": 45.0,  "default": 24.0,  "step": 0.1,
                                 "help": "Indeks Massa Tubuh. Normal 18.5-24.9."},
    "Age":                      {"label": "Usia",                  "unit": "tahun", "min": 18.0,  "max": 100.0, "default": 45.0,  "step": 1.0,
                                 "help": "Usia pasien."},
    "Sleep Hours Per Day":      {"label": "Jam Tidur per Hari",    "unit": "jam",   "min": 3.0,   "max": 12.0,  "default": 7.0,   "step": 0.5,
                                 "help": "Rata-rata jam tidur per hari. Ideal 7-9 jam."},
    "Exercise Hours Per Week":  {"label": "Jam Olahraga per Minggu","unit": "jam",  "min": 0.0,   "max": 20.0,  "default": 3.0,   "step": 0.5,
                                 "help": "Total jam olahraga per minggu."},
    "Sedentary Hours Per Day":  {"label": "Jam Duduk per Hari",    "unit": "jam",   "min": 0.0,   "max": 16.0,  "default": 6.0,   "step": 0.5,
                                 "help": "Jam aktivitas sedentari (duduk) per hari."},
    "Physical Activity Days Per Week": {"label": "Hari Aktif per Minggu", "unit": "hari", "min": 0.0, "max": 7.0, "default": 3.0, "step": 1.0,
                                 "help": "Jumlah hari aktivitas fisik per minggu."},
    "Stress Level":             {"label": "Tingkat Stres",         "unit": "1-10",  "min": 1.0,   "max": 10.0,  "default": 5.0,   "step": 1.0,
                                 "help": "Skala stres 1 (rendah) - 10 (tinggi)."},
    "Income":                   {"label": "Pendapatan Tahunan",    "unit": "USD",   "min": 20000.0,"max": 300000.0,"default": 150000.0,"step": 1000.0,
                                 "help": "Pendapatan tahunan (fitur dari dataset)."},
}

# Kolom biner (Yes/No) — label yang ramah pengguna
BINARY_CONFIG = {
    "Diabetes":                {"label": "Diabetes",                 "help": "Apakah pasien memiliki diabetes?"},
    "Obesity":                 {"label": "Obesitas",                 "help": "Apakah pasien mengalami obesitas?"},
    "Alcohol Consumption":     {"label": "Konsumsi Alkohol",         "help": "Apakah pasien mengonsumsi alkohol?"},
    "Smoking":                 {"label": "Merokok",                  "help": "Apakah pasien merokok?"},
    "Family History":          {"label": "Riwayat Keluarga",         "help": "Ada riwayat penyakit jantung di keluarga?"},
    "Previous Heart Problems": {"label": "Riwayat Masalah Jantung",  "help": "Pernah punya masalah jantung sebelumnya?"},
    "Medication Use":          {"label": "Penggunaan Obat",          "help": "Sedang menggunakan obat rutin?"},
}

# Kolom kategorikal khusus (bukan biner)
CATEGORICAL_OPTIONS = {
    "Sex":  ["Male", "Female"],
    "Diet": ["Average", "Healthy", "Unhealthy"],
}

# ============================================================
# PENGELOMPOKAN FITUR UNTUK TAMPILAN UI
# ============================================================

HISTORY_FEATURES = [
    "Diabetes",
    "Obesity",
    "Alcohol Consumption",
]

LIFESTYLE_FEATURES = [
    "Sleep Hours Per Day",
    "Exercise Hours Per Week",
]

CLINICAL_FEATURES = [
    "Systolic",
    "Diastolic",
    "Cholesterol",
    "Triglycerides",
]

ADDITIONAL_FEATURES = [
    "Income",
]

def build_numeric_widget(col):
    """Bangun number_input untuk kolom numerik."""
    cfg = NUMERIC_CONFIG.get(col, {
        "label": col, "unit": "", "min": 0.0, "max": 1000.0, "default": 0.0, "step": 1.0, "help": ""
    })
    label = f"{cfg['label']} ({cfg['unit']})" if cfg["unit"] else cfg["label"]
    return st.number_input(
        label,
        min_value=cfg["min"], max_value=cfg["max"],
        value=cfg["default"], step=cfg["step"],
        help=cfg.get("help", ""),
    )


def build_nominal_widget(col):
    """Bangun widget untuk kolom nominal (biner Yes/No atau kategorikal)."""
    if col in CATEGORICAL_OPTIONS:
        # Kategorikal seperti Sex, Diet
        label = col if col not in BINARY_CONFIG else BINARY_CONFIG[col]["label"]
        return st.selectbox(label, CATEGORICAL_OPTIONS[col],
                            help=f"Pilih {col.lower()}")
    else:
        # Biner 0/1 → tampilkan Yes/No, kirim string "0"/"1"
        cfg = BINARY_CONFIG.get(col, {"label": col, "help": ""})
        choice = st.radio(
            cfg["label"], ["Tidak", "Ya"],
            horizontal=True, help=cfg.get("help", ""),
        )
        # PENTING: kirim string "0"/"1" (konsisten dengan training .astype(str))
        return "1" if choice == "Ya" else "0"


# ============================================================
# MAIN APP
# ============================================================
def main():
    # --- Header ---
    st.markdown('<div class="main-title">🫀 Heart Attack Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Prediksi risiko serangan jantung berbasis machine learning</div>', unsafe_allow_html=True)

    # --- Load artifacts ---
    if not os.path.exists("artifacts/best_pipeline.pkl"):
        st.error(
            "❌ File `artifacts/best_pipeline.pkl` tidak ditemukan.\n\n"
            "Jalankan notebook (Section 5) terlebih dahulu untuk menghasilkan artifacts, "
            "lalu letakkan folder `artifacts/` di direktori yang sama dengan `app.py`."
        )
        st.stop()

    try:
        pipeline, metadata = load_artifacts()
    except Exception as e:
        st.error(f"❌ Gagal memuat artifacts: {e}")
        st.stop()

    # --- Info model ---
    metrics = metadata.get("metrics", {})
    badges = f'<span class="metric-badge">Model: {metadata.get("model_name", "?")}</span>'
    if "F1" in metrics:
        badges += f'<span class="metric-badge">F1: {metrics["F1"]:.3f}</span>'
    if "Recall" in metrics:
        badges += f'<span class="metric-badge">Recall: {metrics["Recall"]:.3f}</span>'
    st.markdown(badges, unsafe_allow_html=True)

    st.markdown("---")

    # --- Ambil daftar kolom dari metadata ---
    input_columns = metadata["input_columns"]
    nominal_cols = metadata.get("nominal_cols", [])
    numeric_cols = metadata.get("numeric_cols", [])

    st.markdown('<div class="section-header">📋 Masukkan Data Pasien</div>', unsafe_allow_html=True)
    st.caption(f"Model membutuhkan {len(input_columns)} data. Isi sesuai kondisi pasien.")

    # --- Bangun form, kelompokkan numerik & nominal dalam 2 kolom ---
    # ============================================================
    # FORM INPUT
    # ============================================================

    input_data = {}

    # ============================
    # RIWAYAT KESEHATAN
    # ============================
    history_cols = [c for c in HISTORY_FEATURES if c in input_columns]

    if history_cols:
        st.markdown("### 🩺 Riwayat Kesehatan")

        cols = st.columns(len(history_cols))

        for i, col in enumerate(history_cols):
            with cols[i]:
                input_data[col] = build_nominal_widget(col)

    # ============================
    # GAYA HIDUP
    # ============================
    lifestyle_cols = [c for c in LIFESTYLE_FEATURES if c in input_columns]

    if lifestyle_cols:
        st.markdown("### 🏃 Gaya Hidup")

        cols = st.columns(2)

        for i, col in enumerate(lifestyle_cols):
            with cols[i]:
                input_data[col] = build_numeric_widget(col)

    # ============================
    # PEMERIKSAAN KLINIS
    # ============================
    clinical_cols = [c for c in CLINICAL_FEATURES if c in input_columns]

    if clinical_cols:
        st.markdown("### 🩺 Pemeriksaan Klinis")

        cols = st.columns(2)

        for i, col in enumerate(clinical_cols):
            with cols[i % 2]:
                input_data[col] = build_numeric_widget(col)

    # ============================
    # INFORMASI TAMBAHAN
    # ============================
    additional_cols = [c for c in ADDITIONAL_FEATURES if c in input_columns]

    if additional_cols:
        st.markdown("### 👤 Informasi Tambahan")

        for col in additional_cols:
            input_data[col] = build_numeric_widget(col)

    st.markdown("---")

    # --- Tombol prediksi ---
    if st.button("🔍 Prediksi Risiko"):
        # Susun DataFrame 1 baris sesuai urutan input_columns
        row = {col: input_data[col] for col in input_columns}
        input_df = pd.DataFrame([row])

        try:
            prediction = pipeline.predict(input_df)[0]
            proba = pipeline.predict_proba(input_df)[0]
        except Exception as e:
            st.error(f"❌ Gagal memprediksi: {e}")
            st.stop()

        risk_prob = proba[1] * 100
        no_risk_prob = proba[0] * 100

        # --- Tampilkan hasil ---
        if prediction == 1:
            st.markdown(
                f'<div class="result-risk">'
                f'<h3 style="color:#E63946;margin:0;">⚠️ Berisiko (Risk)</h3>'
                f'<p style="margin:0.4rem 0 0 0;color:#1D3557;">'
                f'Model memprediksi pasien <b>berisiko</b> mengalami serangan jantung '
                f'dengan probabilitas <b>{risk_prob:.1f}%</b>.</p>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-safe">'
                f'<h3 style="color:#28A745;margin:0;">✅ Tidak Berisiko (No Risk)</h3>'
                f'<p style="margin:0.4rem 0 0 0;color:#1D3557;">'
                f'Model memprediksi pasien <b>tidak berisiko</b> '
                f'dengan probabilitas <b>{no_risk_prob:.1f}%</b>.</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        # --- Bar probabilitas ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Probabilitas:**")
        c1, c2 = st.columns(2)
        c1.metric("No Risk", f"{no_risk_prob:.1f}%")
        c2.metric("Risk", f"{risk_prob:.1f}%")
        st.progress(risk_prob / 100)

    # --- Disclaimer ---
    st.markdown(
        '<div class="disclaimer">'
        '⚠️ <b>Disclaimer:</b> Model ini dilatih pada dataset <b>synthetic</b> (bukan data pasien nyata) '
        'dan hanya untuk tujuan pembelajaran. Akurasi terbatas (~50-65%). '
        'Hasil prediksi <b>TIDAK boleh</b> dijadikan dasar diagnosis medis. '
        'Konsultasikan dengan tenaga medis profesional untuk pemeriksaan yang sebenarnya.'
        '</div>',
        unsafe_allow_html=True
    )

    # --- Footer ---
    st.markdown(
        '<p style="text-align:center;color:#ADB5BD;font-size:0.8rem;margin-top:2rem;">'
        'Group 5 — Pandas · Data Science Batch 60 · Digital Skola</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
