import streamlit as st
import time
from voice_recorder import record_voice
from predictions import make_predictions

st.title("EMOSPHERE")
image = st.image(r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\images\image.png", caption='Real-Time Emotion Recognition from Voice', use_column_width=True)
duration = st.slider("Select Recording Duration (seconds)", min_value=1, max_value=60, value=5)
model_type = st.selectbox("Choose Model for Prediction", ("LSTM", "CNN"))

if st.button("Record and Analyze"):
    with st.spinner("Recording..."):
        filename = "myvoice.wav"
        record_voice(duration=duration, filename=filename)
        st.success("Recording complete!")
    
    with st.spinner("Analyzing..."):
        emotion = make_predictions(filename, model_type=model_type)
        st.success(f"Detected Emotion: {emotion}")
