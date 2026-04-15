import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = "data"
MODEL_PATH   = "snore_model.pkl"
SCALER_PATH  = "scaler.pkl"
LABELS       = {"0": "Non-Snoring", "1": "Snoring", "2": "Gasping"}
DURATION     = 3
SR           = 22050
N_MFCC       = 40
RANDOM_STATE = 42

# ── Feature Extraction ────────────────────────────────────────────────────────
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SR, duration=DURATION)

        # Pad short clips to ensure consistent length
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
        print(f"  ⚠ Skipped {file_path}: {e}")
        return None

# ── Load Dataset ──────────────────────────────────────────────────────────────
print("📂 Loading dataset...")
X, y = [], []

for label, name in LABELS.items():
    folder = os.path.join(DATA_PATH, label)
    if not os.path.isdir(folder):
        print(f"  ⚠ Folder not found: {folder}")
        continue

    files = [f for f in os.listdir(folder) if f.lower().endswith(".wav")]
    print(f"  Class {label} ({name}): {len(files)} files")

    for file in files:
        features = extract_features(os.path.join(folder, file))
        if features is not None:
            X.append(features)
            y.append(int(label))

X = np.array(X)
y = np.array(y)
print(f"\n✅ Dataset ready — {len(X)} samples, {X.shape[1]} features each")

# ── Class Imbalance ───────────────────────────────────────────────────────────
class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
weight_dict   = dict(enumerate(class_weights))
print(f"\n⚖ Class weights: { {LABELS[str(k)]: round(v, 2) for k, v in weight_dict.items()} }")

# ── Train / Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ── Scaling ───────────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Model ─────────────────────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=4,
    class_weight=weight_dict,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

# ── Cross-Validation ──────────────────────────────────────────────────────────
print("\n🔄 Running 5-fold cross-validation...")
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
print(f"   CV F1 scores : {np.round(cv_scores, 3)}")
print(f"   Mean ± Std   : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── Train + Evaluate ──────────────────────────────────────────────────────────
print("\n🏋 Training final model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n📊 Test Set Results:")
print(classification_report(y_test, y_pred, target_names=list(LABELS.values())))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(model,  MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"\n💾 Model  saved → {MODEL_PATH}")
print(f"💾 Scaler saved → {SCALER_PATH}")
print("✅ Done!")