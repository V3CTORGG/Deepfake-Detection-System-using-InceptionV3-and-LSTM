# 🎭 Deeflyzer: Hybrid Deepfake Detection System

Deeflyzer is an AI-powered system designed to detect deepfake content in both **videos** and **audio** files. Using a combination of **CNN-based deep learning models** for video analysis and **Wav2Vec 2.0** for audio deepfake detection, Deeflyzer provides an efficient and reliable approach to identifying manipulated media. 🎥🔊

## 🚀 Features
- **🎬 Video Deepfake Detection:** Extracts frames, detects faces, and classifies them as real or fake.
- **🎤 Audio Deepfake Detection:** Analyzes speech patterns to detect AI-generated or manipulated audio.
- **📸 Fake Face Visualization:** Displays identified fake faces for better interpretability.
- **💻 Streamlit-based UI:** Simple and interactive web interface for easy deepfake analysis.

## 🛠️ Technologies Used
- **TensorFlow** 🧠 (For deepfake classification)
- **Torch & Torchaudio** 🔥 (For audio deepfake detection)
- **Wav2Vec 2.0** 🎙️ (Advanced speech processing)
- **MTCNN** 🏷️ (For face detection in frames)
- **OpenCV** 📷 (For frame extraction and processing)
- **Streamlit** 🌐 (For interactive UI)

## 📥 Installation
```sh
# Clone the repository
git clone https://github.com/yourusername/deeflyzer.git
cd deeflyzer

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Usage
Run the application using:
```sh
streamlit run app.py
```

### Video Deepfake Detection 🎬
1. Upload a video file (.mp4, .avi, .mkv, etc.).
2. Click on "Classify Video" to analyze the content.
3. View fake face detection results and classification.

### Audio Deepfake Detection 🎤
1. Upload an audio file (.wav format supported).
2. Click on "Classify Audio" to check if it’s fake or real.

## 🖼️ Sample Output
Fake faces detected in the video will be displayed along with classification results. ✅

## 🏆 Contribution
Feel free to open issues or submit pull requests! 🚀

## 📜 License
This project is licensed under the **MIT License**. 📄

---
_Developed with ❤️ to combat digital misinformation!_ 💡

