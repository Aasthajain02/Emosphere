"""Tests for `speech_emotion_recognition` package."""
import sys
import pytest
import os

# Ensure the correct path to the `speech_emotion_recognition` package is included
package_path = r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition"
sys.path.insert(0, package_path)

def test_invalid_path():
    from preprocessing import extract_features

    path = 25
    with pytest.raises(TypeError):
        extract_features(path)

if __name__ == "__main__":
    # This section is to debug path issues
    print("Current Working Directory:", os.getcwd())
    print("System Path:", sys.path)
    test_invalid_path()
