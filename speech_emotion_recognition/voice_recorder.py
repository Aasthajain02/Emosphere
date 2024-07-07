# voice_recorder.py

import sounddevice as sd
from scipy.io.wavfile import write


def record_voice(duration, filename):
    """Record voice for a specified duration and save it as a WAV file."""
    fs = 44100  # Sample rate
    # channels = 2  # Stereo
    print(f"Recording voice for {duration} seconds...")

    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished

    write(filename, fs, myrecording)
    print(f"Voice recording saved to {filename}")
