# 💤 Snore & Sleep Apnea Detection

A simple machine learning project that detects **non-snoring, snoring, and gasping (apnea-related)** sounds from audio files.

---

## 🚀 Features

* Classifies audio into:

  * `0` → Non-snoring (noise)
  * `1` → Snoring
  * `2` → Gasping (apnea indicator)
* Uses a Random Forest model
* Handles class imbalance effectively
* Saves trained model and scaler for reuse

---

## 📂 Project Structure

```
snore-detection/
│
├── data/
│   ├── 0/   # Non-snoring sounds
│   ├── 1/   # Snoring sounds
│   └── 2/   # Gasping sounds
│
├── train_apnea_model.py   # Training script
├── predict.py             # Predict on audio file
├── snore_model.pkl        # Trained model
├── scaler.pkl             # Feature scaler
└── README.md
```

---

## ▶️ Usage

### Train the model

```bash
python train_apnea_model.py
```

### Predict on an audio file

```bash
python predict.py
```

---

## 🎯 Output Labels

| Label | Meaning         |
| ----- | --------------- |
| 0     | Normal / Noise  |
| 1     | Snoring         |
| 2     | Gasping (Apnea) |

---

## ⚠️ Note

This is a **prototype project** intended for learning and experimentation.
It is **not a medical diagnostic tool**.

---

## 👩‍💻 Author

**Ananya**
Aspiring Software Developer

---

## 🙏 Credits

* Audio datasets sourced from publicly available platforms (e.g., Kaggle: * T. H. Khan, "A deep learning model for snoring detection and vibration notification using a smart wearable gadget," Electronics, vol. 8, no. 9, article. 987, ISSN 2079-9292, 2019.
, online recordings, YouTube)
* Built using Python libraries: `librosa`, `scikit-learn`, `numpy`
* Developed through self-learning, experimentation, and open resources

---
