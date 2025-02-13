import os
import cv2
import torch
import numpy as np
import tensorflow as tf
import streamlit as st
import torchaudio
from mtcnn import MTCNN
from PIL import Image
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torchaudio.transforms as transforms
from pydub import AudioSegment

# Paths
frame_output_folder = "frames"
face_output_folder = "faces"
deepfake_model_path = "1.h5"

# Load models
video_model = tf.keras.models.load_model(deepfake_model_path, compile=False)
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
audio_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h", num_labels=2).to("cuda" if torch.cuda.is_available() else "cpu")

IMG_WIDTH, IMG_HEIGHT = 224, 224

def extract_frames(video_path):
    os.makedirs(frame_output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frames = []
    for i in range(7):
        success, frame = cap.read()
        if success:
            frame_path = os.path.join(frame_output_folder, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
    cap.release()
    return frames

def detect_faces():
    os.makedirs(face_output_folder, exist_ok=True)
    detector = MTCNN()
    face_images = []
    face_paths = []
    for file_name in os.listdir(frame_output_folder):
        frame_path = os.path.join(frame_output_folder, file_name)
        image = cv2.imread(frame_path)
        faces = detector.detect_faces(image)
        for i, face in enumerate(faces):
            x, y, w, h = face['box']
            face_img = image[max(0, y):y+h, max(0, x):x+w]
            face_path = os.path.join(face_output_folder, f"{file_name}_face_{i}.jpg")
            cv2.imwrite(face_path, face_img)
            face_images.append(face_img)
            face_paths.append(face_path)
    return face_paths

def classify_faces():
    face_files = os.listdir(face_output_folder)
    face_images = []
    face_data = []  # Store file paths and classifications
    
    for file_name in face_files:
        img_path = os.path.join(face_output_folder, file_name)
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = tf.image.convert_image_dtype(img, tf.float32)
        face_images.append(img)
        face_data.append((file_name, img_path))  # Store image path
    
    if not face_images:
        return [], []

    face_images = np.array(face_images)
    predictions = video_model.predict(face_images)
    labels = ['Fake' if p <= 0.5 else 'Real' for p in predictions]

    fake_faces = [img_path for (file_name, img_path), label in zip(face_data, labels) if label == 'Fake']
    return labels, fake_faces

def preprocess_audio(file_path):
    # Convert non-wav files to wav
    if not file_path.lower().endswith(".wav"):
        wav_path = file_path.rsplit(".", 1)[0] + ".wav"
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")
        file_path = wav_path  # Use the converted file

    # Load the wav file
    audio, sample_rate = torchaudio.load(file_path)

    # Resample if necessary
    if sample_rate != 16000:
        audio = transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)

    return audio.squeeze().numpy()

def classify_audio(file_path):
    audio = preprocess_audio(file_path)
    inputs = audio_processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        logits = audio_model(input_values).logits
    return "Real" if torch.argmax(logits[0]).item() == 0 else "Fake"

st.title("Deeflyzer: Hybrid Deepfake Detection System")
st.sidebar.title("Options")

choice = st.sidebar.selectbox("Select Mode", ["Video Deepfake Detection", "Audio Deepfake Detection"])

if choice == "Video Deepfake Detection":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mkv", "mov"])
    if uploaded_file is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.video("temp_video.mp4")
        if st.button("Classify Video"):
            extract_frames("temp_video.mp4")
            detect_faces()
            face_labels, fake_faces = classify_faces()
            fake_count = face_labels.count('Fake')
            real_count = face_labels.count('Real')
            result = "Fake" if fake_count > real_count else "Real"
            st.success(f"The video is classified as: {result}")
            st.write(f"Real Faces: {real_count}, Fake Faces: {fake_count}")

            if fake_faces:
                st.subheader("Fake Faces Detected:")
                cols = st.columns(min(4, len(fake_faces)))
                for i, face_path in enumerate(fake_faces):
                    with cols[i % len(cols)]:
                        st.image(face_path, caption=f"Fake Face {i+1}", use_container_width=True)

elif choice == "Audio Deepfake Detection":
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
    if uploaded_file is not None:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.audio("temp_audio.wav")
        if st.button("Classify Audio"):
            result = classify_audio("temp_audio.wav")
            st.success(f"The audio is classified as: {result}")
