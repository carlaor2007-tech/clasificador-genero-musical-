FROM python:3.10-alpine

RUN apk add --no-cache gcc musl-dev libffi-dev ffmpeg libsndfile-dev

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app_streamlit.py .
COPY model_weights.npz .
COPY genre_names.npy .
COPY scaler.pkl .
COPY feature_cols.pkl .

EXPOSE 8501

CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]

