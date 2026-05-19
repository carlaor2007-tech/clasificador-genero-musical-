import streamlit as st
import numpy as np
import librosa
import joblib
import os
import tempfile

# ── Configuración de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador de Género Musical",
    page_icon="🎵",
    layout="centered"
)

# ── Carga del modelo y artefactos ─────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    from tensorflow.keras.models import load_model
    modelo      = load_model("modelo_genero_musical.h5")
    generos     = np.load("genre_names.npy", allow_pickle=True)
    scaler      = joblib.load("scaler.pkl")      if os.path.exists("scaler.pkl")      else None
    feat_cols   = joblib.load("feature_cols.pkl") if os.path.exists("feature_cols.pkl") else None
    return modelo, generos, scaler, feat_cols

model, genre_names, SCALER, FEATURE_COLS = cargar_modelo()

# ── Extracción de features ────────────────────────────────────────────────────
def extract_features(path):
    y, sr = librosa.load(path, mono=True)
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    harm  = librosa.effects.harmonic(y)
    perc  = librosa.effects.percussive(y)

    feats = {
        "chroma_stft_mean":        float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr))),
        "chroma_stft_var":         float(np.var(librosa.feature.chroma_stft(y=y, sr=sr))),
        "rms_mean":                float(np.mean(librosa.feature.rms(y=y))),
        "rms_var":                 float(np.var(librosa.feature.rms(y=y))),
        "spectral_centroid_mean":  float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        "spectral_centroid_var":   float(np.var(librosa.feature.spectral_centroid(y=y, sr=sr))),
        "spectral_bandwidth_mean": float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
        "spectral_bandwidth_var":  float(np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
        "rolloff_mean":            float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
        "rolloff_var":             float(np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))),
        "zero_crossing_rate_mean": float(np.mean(librosa.feature.zero_crossing_rate(y))),
        "zero_crossing_rate_var":  float(np.var(librosa.feature.zero_crossing_rate(y))),
        "harmony_mean":            float(np.mean(harm)),
        "harmony_var":             float(np.var(harm)),
        "perceptr_mean":           float(np.mean(perc)),
        "perceptr_var":            float(np.var(perc)),
        "tempo":                   float(librosa.beat.tempo(y=y, sr=sr)[0]),
    }
    for i in range(1, 21):
        feats[f"mfcc{i}_mean"] = float(np.mean(mfcc[i-1]))
        feats[f"mfcc{i}_var"]  = float(np.var(mfcc[i-1]))

    cols = FEATURE_COLS if FEATURE_COLS is not None else list(feats.keys())
    vec  = np.array([feats.get(c, 0.0) for c in cols], dtype=np.float32).reshape(1, -1)
    if SCALER is not None:
        vec = SCALER.transform(vec)
    return vec

# ── Interfaz ──────────────────────────────────────────────────────────────────
st.title("🎵 Sistema de Identificación del Género Musical")
st.markdown(
    "Sube un archivo de audio **.wav** y la red neuronal identificará "
    "su género musical usando coeficientes **MFCC**."
)

st.markdown("---")

audio_file = st.file_uploader("Selecciona un archivo .wav", type=["wav"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    with st.spinner("Analizando el audio..."):
        # Guardar en archivo temporal para que librosa lo lea
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        try:
            vec   = extract_features(tmp_path)
            probs = model.predict(vec, verbose=0)[0]
            idx   = int(np.argmax(probs))
            genre = genre_names[idx]
            conf  = float(probs[idx]) * 100
        except Exception as e:
            st.error(f"Error al procesar el audio: {e}")
            st.stop()
        finally:
            os.unlink(tmp_path)

    st.markdown("---")
    st.subheader("Resultado")

    col1, col2 = st.columns(2)
    col1.metric("Género predicho", genre.upper())
    col2.metric("Confianza", f"{conf:.1f}%")

    st.markdown("#### Probabilidades por género")
    prob_dict = {g: float(p) for g, p in zip(genre_names, probs)}
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))

    for g, prob in sorted_probs.items():
        pct = prob * 100
        color = "🟢" if g == genre else "⚪"
        st.progress(prob, text=f"{color} **{g}** — {pct:.1f}%")

st.markdown("---")
st.caption(
    "Dataset: GTZAN Genre Collection (Data-2) · 1.000 canciones · 10 géneros · "
    "Red neuronal feedforward con features MFCC · "
    "Laura Monzó · Carla Ortega · Alicia Carmona"
)
