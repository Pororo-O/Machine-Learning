from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Lokasi folder model: Tugas/model (sejajar dengan folder code)
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

st.set_page_config(page_title="Prediksi Skor IMDB Netflix", page_icon="🎬")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "model_linear_regression.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    features = joblib.load(MODEL_DIR / "features.joblib")
    return model, scaler, features


st.title("🎬 Prediksi Skor IMDB Film/Serial Netflix")
st.write(
    "Aplikasi ini memakai model Regresi Linear untuk memperkirakan skor IMDB "
    "berdasarkan tahun rilis, durasi, dan jumlah suara."
)

try:
    model, scaler, features = load_artifacts()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

col1, col2, col3 = st.columns(3)
release_year = col1.number_input("Tahun rilis", 1900, 2030, 2020, step=1)
runtime = col2.number_input("Durasi (menit)", 1, 600, 100, step=1)
imdb_votes = col3.number_input("Jumlah suara IMDB", 0, 5_000_000, 50_000, step=1000)

if st.button("Prediksi", type="primary"):
    data = pd.DataFrame(
        [[release_year, runtime, imdb_votes]],
        columns=features,
    )
    pred = float(model.predict(scaler.transform(data))[0])
    pred = min(max(pred, 0.0), 10.0)  # skor IMDB hanya 0-10
    st.metric("Prediksi skor IMDB", f"{pred:.2f} / 10")
    st.caption(
        "Catatan: model hanya memakai 3 fitur numerik sehingga akurasinya "
        "terbatas (lihat skor R2 pada notebook)."
    )