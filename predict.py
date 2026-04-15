import librosa
import numpy as np
import joblib
import sys
import os

# -- Config (must match training) ----------------------------------------------
MODEL_PATH   = "snore_model.pkl"
SCALER_PATH  = "scaler.pkl"
DURATION     = 3
SR           = 22050
N_MFCC       = 40
LABELS       = {0: "Non-Snoring", 1: "Snoring", 2: "Gasping"}

# -- Feature Extraction (same as training) -------------------------------------
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SR, duration=DURATION)

        target_len = DURATION * SR
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))

        mfcc       = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        delta_mfcc = librosa.feature.delta(mfcc)
        chroma     = librosa.feature.chroma_stft(y=audio, sr=sr)
        contrast   = librosa.feature.spectral_contrast(y=audio, sr=sr)
        rms        = librosa.feature.rms(y=audio)
        zcr        = librosa.feature.zero_crossing_rate(y=audio)

        return np.concatenate([
            np.mean(mfcc, axis=1),       np.std(mfcc, axis=1),
            np.mean(delta_mfcc, axis=1), np.std(delta_mfcc, axis=1),
            np.mean(chroma, axis=1),     np.std(chroma, axis=1),
            np.mean(contrast, axis=1),
            [np.mean(rms)],
            [np.mean(zcr)],
        ])

    except Exception as e:
        print(f"ERROR reading file: {e}")
        return None

# -- Load Model & Scaler -------------------------------------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    print("ERROR: Model or scaler not found. Run train_apnea_model.py first.")
    sys.exit(1)

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -- Get File Path -------------------------------------------------------------
if len(sys.argv) < 2:
    file_path = input("Enter path to .wav file: ").strip()
else:
    file_path = sys.argv[1]

if not os.path.exists(file_path):
    print(f"ERROR: File not found: {file_path}")
    sys.exit(1)

# -- Predict -------------------------------------------------------------------
print(f"\nAnalyzing: {file_path}")
features = extract_features(file_path)

if features is None:
    print("ERROR: Could not extract features from this file.")
    sys.exit(1)

features   = scaler.transform([features])
prediction = model.predict(features)[0]
confidence = model.predict_proba(features)[0]

print(f"\nPrediction  : {LABELS[prediction]} (Class {prediction})")
print(f"Confidence  :")
for cls, prob in enumerate(confidence):
    bar = "#" * int(prob * 30)
    print(f"   {LABELS[cls]:<15} {prob*100:5.1f}%  {bar}")