from flask import Flask, request, jsonify
import io
import os
import numpy as np
import librosa
import tensorflow as tf
import keras
import joblib

app = Flask(__name__)

# Initialize model and scaler to None
model = None
scaler = None

def load_model_and_scaler():
    global model, scaler
    # Set the model and scaler paths within the Docker container
    model_path = '/app/audio_classifier/1'
    scaler_path = '/app/scaler.joblib'
    
    if model is None:
        model = keras.models.load_model(model_path)
    if scaler is None:
        scaler = joblib.load(scaler_path)

@app.route('/ping', methods=['GET'])
def ping():
    # Check if the model was loaded correctly
    load_model_and_scaler()
    status = 200 if model and scaler else 404
    return '', status

# Function to pad the audio
def pad_audio_to_length(audio, target_length, sr):
    current_length = len(audio)
    target_length_samples = int(target_length * sr)
    if current_length < target_length_samples:
        pad_length = target_length_samples - current_length
        padded_audio = np.pad(audio, (0, pad_length), mode='constant')
    else:
        padded_audio = audio
    return padded_audio

# Function to preprocess and extract features
def extract_features(audio, sr, target_length=4.0, n_mfcc=13, n_fft=2048, hop_length=512):
    audio = pad_audio_to_length(audio, target_length, sr)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfccs_mean = np.mean(mfccs, axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio, hop_length=hop_length))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=22050, n_fft=n_fft, hop_length=hop_length))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=22050, n_fft=n_fft, hop_length=hop_length))
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=22050, n_fft=n_fft, hop_length=hop_length), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=22050, n_fft=n_fft, hop_length=hop_length), axis=1)
    mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=audio, sr=22050, n_fft=n_fft, hop_length=hop_length))

    # Calculate onset envelope and temporal features
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    attack_time = np.argmax(onset_env) * hop_length / sr
    decay_time = len(audio) / sr - attack_time

    features_vector = np.concatenate([
        mfccs_mean,
        [zcr, spectral_centroid, spectral_rolloff, mel_spectrogram],
        spectral_contrast,
        chroma,
        [attack_time, decay_time]
    ])
    return features_vector.reshape(1, -1)

@app.route('/invocations', methods=['POST'])
def predict():
    if request.content_type == 'audio/wav':
        audio_buffer = io.BytesIO(request.data)
        audio, sr = librosa.load(audio_buffer, sr=None)
        features = extract_features(audio, sr)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        predicted_class = (prediction > 0.4).astype(int)
        message = "Threat detected." if predicted_class[0][0] == 1 else "No threat detected."
        return jsonify({'message': message})
    else:
        raise ValueError("Unsupported content type: " + request.content_type)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))