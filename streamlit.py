import streamlit as st
import numpy as np
import librosa
from pydub import AudioSegment
from keras.models import load_model

def preprocess_audio(file):
    _, sr = librosa.load(file, sr=None)
    raw_audio = AudioSegment.from_wav(file)
    samples = np.array(raw_audio.get_array_of_samples(), dtype='float32')
    trimmed, _ = librosa.effects.trim(samples, top_db=25)
    padded = np.pad(trimmed, (0, len(trimmed)), 'constant')
    return padded, sr

st.title('Emotion Prediction from Audio')

# Debugging logs
st.write('Loading model...')
try:
    model = load_model("model2.keras")
    st.write('Model loaded successfully.')
except Exception as e:
    st.write(f'Error loading model: {e}')

# File uploader for audio file
uploaded_file = st.file_uploader('Upload an audio file', type=['wav'])

if uploaded_file is not None:
    st.write('File uploaded, preprocessing...')
    try:
        y, sr = preprocess_audio(uploaded_file)
        FRAME_LENGTH = 2048
        HOP_LENGTH = 512

        zcr = [librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)]
        rms = [librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)]
        mfccs = [librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)]

        input = np.concatenate((
                np.swapaxes(zcr, 1, 2), 
                np.swapaxes(rms, 1, 2), 
                np.swapaxes(mfccs, 1, 2)),
                axis = 2
        )

        input = input.astype('float32')
        st.write('Preprocessing done, making predictions...')
        prediction = model.predict(input)
        predicted_class = np.argmax(prediction, axis=1)[0]
        emotion = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust']

        st.write(f'This is prob {prediction}')
        st.write(f'Predicted Emotion: {emotion[predicted_class]}')
    except Exception as e:
        st.write(f'Error processing file: {e}')
