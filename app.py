"""
Dashboard Deteksi Dini Risiko Penurunan Fokus Akademik
Mahasiswa Pengguna Short-Form Video TikTok Berbasis Machine Learning

Tahap 9 - Perancangan dan Implementasi Dashboard
Jalankan dengan: streamlit run app.py
"""

import altair as alt
import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Dashboard Deteksi Risiko TikTok",
    page_icon="📱",
    layout="centered"
)

# ---------- Styling kustom ----------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #FFF5F0 0%, #FFFFFF 400px);
    }
    .custom-header {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8C69 100%);
        padding: 2rem 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(255,75,75,0.25);
    }
    .custom-header h1 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
    }
    .custom-header p {
        color: white;
        opacity: 0.9;
        margin-top: 0.5rem;
        font-size: 0.95rem;
    }
    .stButton>button {
        border-radius: 12px;
        font-weight: 700;
        padding: 0.7rem 1rem;
        font-size: 1.05rem;
        border: none;
        box-shadow: 0 4px 12px rgba(255,75,75,0.3);
    }
    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #FFD6D6;
        background-color: #FFFBFA;
    }
    section[data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="custom-header">
    <h1>📱 Dashboard Deteksi Risiko Fokus Akademik</h1>
    <p>Universitas Negeri Medan — Program Studi Ilmu Komputer</p>
</div>
""", unsafe_allow_html=True)

# ---------- Load model ----------
@st.cache_resource
def load_model():
    return joblib.load("risk_model.joblib")

bundle = load_model()
model = bundle["model"]
scaler = bundle["scaler"]
feature_cols = bundle["feature_cols"]

RISK_COLORS = {"Ringan": "#4CAF50", "Sedang": "#FF9800", "Berat": "#F44336"}
RISK_DESC = {
    "Ringan": "Pola konsumsi TikTok kamu masih dalam batas wajar. Tetap jaga keseimbangan waktu belajar dan hiburan.",
    "Sedang": "Ada indikasi pola konsumsi TikTok mulai mengganggu fokus akademik. Pertimbangkan membatasi waktu penggunaan, terutama saat jam belajar.",
    "Berat": "Pola konsumsi TikTok kamu berisiko tinggi mengganggu fokus akademik. Disarankan untuk mulai menerapkan manajemen waktu digital yang lebih ketat.",
}

FEATURE_IMPORTANCE = {
    "Durasi penggunaan": 0.226,
    "Buka saat mengerjakan tugas": 0.224,
    "Menunda tidur": 0.150,
    "Frekuensi membuka": 0.130,
    "Sulit berhenti scrolling": 0.126,
    "Buka otomatis/tanpa sadar": 0.076,
}

st.divider()

# ---------- Input form ----------
st.subheader("Isi pola konsumsi TikTok kamu")

col1, col2 = st.columns(2)
with col1:
    durasi_label = st.selectbox(
        "Rata-rata durasi penggunaan TikTok per hari",
        ["< 1 jam", "1 - 2 jam", "2 - 4 jam", "4 - 6 jam", "> 6 jam"],
        index=2,
    )
with col2:
    frekuensi_label = st.selectbox(
        "Berapa kali membuka aplikasi TikTok per hari",
        ["< 5 kali", "5 - 10 kali", "11 - 20 kali", "21 - 30 kali", "> 30 kali"],
        index=1,
    )

durasi_map = {"< 1 jam": 1, "1 - 2 jam": 2, "2 - 4 jam": 3, "4 - 6 jam": 4, "> 6 jam": 5}
frek_map = {"< 5 kali": 1, "5 - 10 kali": 2, "11 - 20 kali": 3, "21 - 30 kali": 4, "> 30 kali": 5}

st.markdown("**Seberapa setuju kamu dengan pernyataan berikut? (1 = Tidak Pernah, 5 = Selalu)**")

pk1 = st.slider("Saya membuka TikTok tanpa sadar/otomatis saat ada waktu luang", 1, 5, 3)
pk2 = st.slider("Saya kesulitan berhenti scrolling TikTok meskipun sudah berniat berhenti", 1, 5, 3)
pk3 = st.slider("Saya membuka TikTok saat sedang mengerjakan tugas kuliah", 1, 5, 3)
pk4 = st.slider("Saya menunda tidur karena masih scrolling TikTok", 1, 5, 3)

st.divider()

# ---------- Predict ----------
if st.button("🔍 Cek Tingkat Risiko Saya", type="primary", use_container_width=True):
    X_input = np.array([[
        durasi_map[durasi_label],
        frek_map[frekuensi_label],
        pk1, pk2, pk3, pk4
    ]])
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    classes = model.classes_

    color = RISK_COLORS.get(pred, "#888")
    st.markdown(
        f"""
        <div style="padding:20px;border-radius:10px;background-color:{color}22;border:2px solid {color};text-align:center;">
            <h2 style="color:{color};margin:0;">Tingkat Risiko: {pred}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.info(RISK_DESC.get(pred, ""))

    st.write("**Probabilitas tiap kategori:**")

    # Kotak angka persen
    cols = st.columns(len(classes))
    for c, k, p in zip(cols, classes, proba):
        c.metric(k, f"{p*100:.1f}%")

    # Grafik batang (sumbu Y dikunci 0-100%, urutan batang tetap)
    proba_df = pd.DataFrame({"Kategori": classes, "Probabilitas": proba})
    chart = alt.Chart(proba_df).mark_bar().encode(
        x=alt.X("Kategori", sort=["Ringan", "Sedang", "Berat"], axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Probabilitas", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color(
            "Kategori",
            scale=alt.Scale(domain=list(RISK_COLORS.keys()), range=list(RISK_COLORS.values())),
            legend=None,
        ),
        tooltip=[alt.Tooltip("Kategori"), alt.Tooltip("Probabilitas", format=".1%")],
    )
    st.altair_chart(chart, use_container_width=True)

st.divider()

# ---------- Feature importance ----------
with st.expander("📊 Fitur apa yang paling berpengaruh ke tingkat risiko?"):
    st.caption("Berdasarkan analisis permutation importance pada model (Tahap 8 penelitian)")
    imp_df = pd.DataFrame(
        {"Fitur": list(FEATURE_IMPORTANCE.keys()), "Tingkat Pengaruh": list(FEATURE_IMPORTANCE.values())}
    ).sort_values("Tingkat Pengaruh", ascending=True)
    st.bar_chart(imp_df.set_index("Fitur"))

st.caption("Dashboard ini dikembangkan sebagai bagian dari penelitian Metodologi Penelitian — Deteksi Dini Risiko Penurunan Fokus Akademik Mahasiswa Pengguna Short-Form Video TikTok Berbasis Machine Learning.")