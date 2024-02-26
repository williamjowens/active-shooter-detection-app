from flask import Flask, render_template, request, jsonify, send_file, session
import numpy as np
import librosa
import tensorflow as tf
import joblib
import os
import secrets
import random

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Load the trained model and the scaler
try:
    model = tf.keras.models.load_model('best_model_final')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
scaler = joblib.load('scaler.joblib')

# Define the path to the audio files
AUDIO_FOLDER_PATH = 'audio'

# Define the function to pad audio
def pad_audio_to_length(audio, target_length, sr):
    current_length = len(audio)
    target_length_samples = int(target_length * sr)
    if current_length < target_length_samples:
        pad_length = target_length_samples - current_length
        padded_audio = np.pad(audio, (0, pad_length), mode='constant')
    else:
        padded_audio = audio
    return padded_audio

# Define the function to extract features
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

@app.route('/')
def index():
    # Render the main page
    return render_template('index.html')

@app.route('/play_random_sound', methods=['GET'])
def play_random_sound():
    # Get a list of audio files
    audio_files = os.listdir(AUDIO_FOLDER_PATH)
    
    if not audio_files:
        return jsonify(result="Error: No audio files available."), 400
    
    # Randomly select an audio file
    selected_file = random.choice(audio_files)
    session['selected_audio'] = selected_file
    
    # Send the audio file for playback
    return send_file(os.path.join(AUDIO_FOLDER_PATH, selected_file), as_attachment=True)

@app.route('/predict', methods=['GET'])
def predict():
    # Retrieve the selected audio file name from the session
    selected_file = session.pop('selected_audio', None)
    if not selected_file:
        return jsonify(result="Error: No audio file selected."), 400
    
    try:
        # Load audio file
        audio_path = os.path.join(AUDIO_FOLDER_PATH, selected_file)
        audio, sr = librosa.load(audio_path, sr=None)

        # Extract features
        features = extract_features(audio, sr)

        # Scale the features
        features_scaled = scaler.transform(features)

        # Predict the class
        prediction = model.predict(features_scaled)
        predicted_class = (prediction > 0.5).astype(int)

        # Determine the message
        if predicted_class == 1:
            message = "Threat detected. Local authorities and emergency services have been contacted."
        else:
            message = "No threat detected."

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        message = "Error: An issue occurred during the analysis."

    # Return the message
    return jsonify(result=message)
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)