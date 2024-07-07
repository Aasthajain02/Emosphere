import os
import time
import joblib
import librosa
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

def extract_file_info():
    DATA_PATH = r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\data"

    df = pd.DataFrame(columns=["file", "gender", "emotion", "intensity"])

    for dirname, _, filenames in os.walk(DATA_PATH):
        for filename in filenames:

            emotion = filename[7]
            if emotion == "1":
                emotion = "neutral"
            elif emotion == "2":
                emotion = "calm"
            elif emotion == "3":
                emotion = "happy"
            elif emotion == "4":
                emotion = "sad"
            elif emotion == "5":
                emotion = "angry"
            elif emotion == "6":
                emotion = "fearful"
            elif emotion == "7":
                emotion = "disgusted"
            elif emotion == "8":
                emotion == "surprised"

            intensity = filename[10]
            if intensity == "1":
                emotion_intensity = "normal"
            elif intensity == "2":
                emotion_intensity = "strong"

            gender = filename[-6:-4]
            if int(gender) % 2 == 0:
                gender = "female"
            else:
                gender = "male"

            df = df._append(
                {
                    "file": filename,
                    "gender": gender,
                    "emotion": emotion,
                    "intensity": emotion_intensity,
                },
                ignore_index=True,
            )

    df.to_csv(r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\features\df_features.csv", index=False)


def extract_features(path, save_dir):
    """This function loops over the audio files,
    extracts the MFCC, and saves X and y in joblib format.
    """
    feature_list = []

    start_time = time.time()
    try:
        for dirpath, _, filenames in os.walk(path):
            for file in filenames:
                try:
                    full_file_path = os.path.join(dirpath, file)
                    y_lib, sample_rate = librosa.load(full_file_path, res_type="kaiser_fast")
                    mfccs = np.mean(
                        librosa.feature.mfcc(y=y_lib, sr=sample_rate, n_mfcc=40).T, axis=0
                    )
                    emotion_label = int(file[7]) - 1  # Assuming emotion is encoded in filename
                    feature_list.append((mfccs, emotion_label))
                except Exception as e:
                    print(f"Error processing file {file}: {str(e)}")
    except Exception as e:
        print(f"Error walking directory {path}: {str(e)}")

    print(f"Data loaded in {time.time() - start_time} seconds.")

    if not feature_list:
        print("Warning: No features extracted. Check input data or processing logic.")

    else:
        X, y = zip(*feature_list)
        X, y = np.asarray(X), np.asarray(y)
        print(f"X shape: {X.shape}, y shape: {y.shape}")

        X_save, y_save = "X.joblib", "y.joblib"
        joblib.dump(X, os.path.join(save_dir, X_save))
        joblib.dump(y, os.path.join(save_dir, y_save))

        return "Preprocessing completed."




def extract_audio_features():
    """This function loops over the audio files,
    extracts four audio feature and saves them in a dataframe.
    """


    df = pd.DataFrame(columns=['chroma','mel_spectrogram'])

    counter=0

    for index,path in enumerate(audio_df.path):
        X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=3,sr=44100,offset=0.5)
            
        chroma = librosa.feature.chroma_stft(y=X, sr=sample_rate)
        chroma = np.mean(chroma, axis = 0)
        df.loc[counter] = [chroma]
        counter=counter+1   

        spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128,fmax=8000) 
        db_spec = librosa.power_to_db(spectrogram)
        log_spectrogram = np.mean(db_spec, axis = 0)
        df.loc[counter] = [log_spectrogram]
        counter=counter+1  

    df_chroma = pd.concat([audio_df,pd.DataFrame(df['chroma'].values.tolist())],axis=1)
    df_chroma = df_combined.fillna(0)
    df_chroma.head()

    df_chroma.to_csv(r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\features\df_chroma.csv", index=0)

def oversample(X, y):
    X = joblib.load(r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\features\X.joblib")  # mfcc
    y = joblib.load(r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\features\y.joblib")
    print(Counter(y))  # {7: 192, 4: 192, 3: 192, 1: 192, 6: 192, 2: 192, 5: 192, 0: 96}

    oversample = RandomOverSampler(sampling_strategy="minority")
    X_over, y_over = oversample.fit_resample(X, y)

    X_over_save, y_over_save = "X_over.joblib", "y_over.joblib"
    joblib.dump(X_over, os.path.join("speech_emotion_recognition/features/", X_over_save))
    joblib.dump(y_over, os.path.join("speech_emotion_recognition/features/", y_over_save))

if __name__ == "__main__":
    print("Extracting file info...")
    extract_file_info()
    print("Extracting audio features...")
    FEATURES = extract_features(path=r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\data",save_dir=r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\features")
    print("Finished extracting audio features.")