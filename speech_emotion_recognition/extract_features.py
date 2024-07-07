import os
import pandas as pd

def extract_file_info(path):
    print(f"Starting to extract file information from: {path}")
    if not os.path.exists(path):
        print(f"Error: The specified path does not exist: {path}")
        return
    actor_folders = os.listdir(path)
    
    file_path=[]
    actor=[]
    emotion=[]
    intensity=[]
    gender=[]
    
    
    for i in actor_folders:
        folder_path = os.path.join(path,i)
        filename = os.listdir(folder_path)
        print(f"Processing file: ")
        for f in filename:
            full_file_path = os.path.join(folder_path, f)
            if os.path.isdir(full_file_path):
                continue
            
            file_path.append(full_file_path)
            parts = f.split('-')
            if len(parts) < 7:
                continue
            
            try:
                emotion.append(int(parts[2]))
                bg = int(parts[-1].split('.')[0])  # Actor ID is the last part before the extension
                actor.append(bg)
                intensity.append(int(parts[3]))
                
                if bg % 2 == 0:
                    gender.append("female")
                else:
                    gender.append("male")
            except ValueError as e:
                print(f"Error processing file {f}: {e}")
    
    audio_df= pd.DataFrame(emotion)
    audio_df= audio_df.replace(
        {
            1:"neutral",
            2:"calm",
            3: "happy",
            4: "sad",
            5: "angry",
            6: "fearful",
            7: "disgusted",
            8: "surprised",
        }
    )
    print("Concatenating gender, emotion, intensity, and actor data...")
    audio_df=pd.concat([pd.DataFrame(gender),audio_df,pd.DataFrame(intensity),pd.DataFrame(actor)],axis=1)
    audio_df.columns=["gender", "emotion", "intensity", "actor"]
    print("Adding file paths to DataFrame...")
    audio_df= pd.concat([audio_df,pd.DataFrame(file_path,columns=["path"])],axis=1)
    print(f"Saving DataFrame to path...")
    output_path = r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\features\df_features_new.csv"


    audio_df.to_csv(output_path, index=False)
    print("File saved successfully.")
    


path= r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\data"
extract_file_info(path)