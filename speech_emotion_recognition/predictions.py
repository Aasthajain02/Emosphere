import numpy as np
import librosa
from tensorflow import keras

def make_predictions(file, model_type="LSTM"):
    if model_type == "CNN":
        model = keras.models.load_model(r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\models\cnn_model.h5")
    elif model_type == "LSTM":
        model = keras.models.load_model(r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\models\lstm_model.h5")
    else:
        raise ValueError("Invalid model type. Choose 'CNN' or 'LSTM'.")

    prediction_data, prediction_sr = librosa.load(
        file, res_type="kaiser_fast", duration=3, sr=22050, offset=0.5
    )

    mfccs = np.mean(librosa.feature.mfcc(y=prediction_data, sr=prediction_sr, n_mfcc=40).T, axis=0)
    x = np.expand_dims(mfccs, axis=1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    predicted_class = np.argmax(predictions, axis=1)[0]

    emotions_dict = {
        0: "neutral",
        1: "calm",
        2: "happy",
        3: "sad",
        4: "angry",
        5: "fearful",
        6: "disgusted",
        7: "surprised"
    }

    return emotions_dict.get(predicted_class, "Unknown")
