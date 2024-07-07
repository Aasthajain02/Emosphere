import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    Flatten,
    Dropout,
    Activation,
    MaxPooling1D,
    BatchNormalization,
    LSTM,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from collections import Counter


def mlp_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(100,),
        solver="adam",
        alpha=0.001,
        shuffle=True,
        verbose=True,
        momentum=0.8,
    )
    mlp_model.fit(X_train, y_train)

    mlp_pred = mlp_model.predict(X_test)
    mlp_accuracy = mlp_model.score(X_test, y_test)
    print("Accuracy: {:.2f}%".format(mlp_accuracy * 100))  # 47.57%

    mlp_clas_report = pd.DataFrame(classification_report(y_test, mlp_pred, output_dict=True)).transpose()
    mlp_clas_report.to_csv(r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\features\mlp_class_report.csv")
    print(classification_report(y_test, mlp_pred))


def lstm_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_lstm = np.expand_dims(X_train, axis=2)
    X_test_lstm = np.expand_dims(X_test, axis=2)
    y_train = to_categorical(y_train)
    y_test=to_categorical(y_test)
    lstm_model = Sequential()
    lstm_model.add(LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True))
    lstm_model.add(LSTM(32))
    lstm_model.add(Dense(32, activation="relu"))
    lstm_model.add(Dropout(0.1))
    lstm_model.add(Dense(8, activation="softmax"))

    lstm_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    lstm_model.summary()

    # train model
    lstm_history = lstm_model.fit(X_train_lstm, y_train, batch_size=32, epochs=100)

    # evaluate model on test set
    test_loss, test_acc = lstm_model.evaluate(X_test_lstm, y_test, verbose=2)
    print("\nTest accuracy:", test_acc)

    # plot accuracy/error for training and validation
    plt.plot(lstm_history.history["loss"])
    plt.title("LSTM model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig(r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\images\lstm_loss.png")
    plt.close()

    # Plot model accuracy
    plt.plot(lstm_history.history["accuracy"])
    plt.title("LSTM model accuracy")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig(r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\images\lstm_accuracy.png")
    plt.close
    model_path = r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\models\lstm_model.h5"
    lstm_model.save(model_path)
    print("Saved trained model at:", model_path)


def cnn_model(X, y):
    """
    This function transforms the X and y features,
    trains a convolutional neural network, and plots the results.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    x_traincnn = np.expand_dims(X_train, axis=2)
    x_testcnn = np.expand_dims(X_test, axis=2)
    y_train = to_categorical(y_train)
    y_test_categorical = to_categorical(y_test)
    model = Sequential()
    model.add(Conv1D(16, 5, padding="same", input_shape=(40, 1)))
    model.add(Activation("relu"))
    model.add(Conv1D(8, 5, padding="same"))
    model.add(Activation("relu"))
    # model.add(Dropout(0.1))  # 0.3
    # model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(8,5,padding="same",))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Activation("softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",  # rmsprop
        metrics=["accuracy"],
    )

    cnn_history = model.fit(
        x_traincnn,
        y_train,
        batch_size=50,  # 100
        epochs=100,  # 50
        validation_data=(x_testcnn,y_test_categorical),
    )

    # Plot model loss and accuracy
    plot_model_metrics(cnn_history)

    # Evaluate the model
    evaluate_model(model, x_testcnn, y_test)

    # Save confusion matrix
    save_confusion_matrix(model, x_testcnn, y_test)

    # Save classification report
    save_classification_report(y_test, model.predict(x_testcnn))

    # Save trained model
    save_trained_model(model)

# def plot_model_summary(model):
#     plot_model(
#         model,
#         to_file=r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\images\cnn_model_summary.png",
#         show_shapes=True,
#         show_layer_names=True,
#     )

def plot_model_metrics(history):
    # Plot model loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("CNN model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"])
    plt.savefig(r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\images\cnn_loss.png")
    plt.close()

    # Plot model accuracy
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("CNN model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"])
    plt.savefig(r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\images\cnn_accuracy.png")
    plt.close()

def evaluate_model(model, x_test, y_test):
    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, to_categorical(y_test), verbose=2)
    print("\nTest accuracy:", test_acc)

def save_confusion_matrix(model, x_test, y_test):
    # Generate confusion matrix
    cnn_pred = model.predict(x_test)
    cnn_pred_classes = np.argmax(cnn_pred, axis=1)
    print("y_true:", y_test)
    print("y_pred:", cnn_pred_classes)
    
    matrix = confusion_matrix(y_test, cnn_pred_classes)
    print(matrix)
    # Plot confusion matrix
    plot_confusion_matrix(matrix)

def plot_confusion_matrix(matrix):
    plt.figure(figsize=(12, 10))
    emotions = [
        "neutral", "calm", "happy", "sad",
        "angry", "fearful", "disgusted", "surprised"
    ]
    cm = pd.DataFrame(matrix)
    ax = sns.heatmap(
        matrix,
        linecolor="white",
        cmap="crest",
        linewidth=1,
        annot=True,
        fmt="",
        xticklabels=emotions,
        yticklabels=emotions,
    )
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title("CNN Model Confusion Matrix", size=20)
    plt.xlabel("Predicted Emotion", size=14)
    plt.ylabel("Actual Emotion", size=14)
    plt.savefig(r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\images\CNN_confusionmatrix.png")
    plt.show()

def save_classification_report(y_true, y_pred):
    # Generate classification report
    y_pred_classes = np.argmax(y_pred, axis=1)
    clas_report = classification_report(y_true, y_pred_classes, output_dict=True)
    clas_report_df = pd.DataFrame(clas_report).transpose()

    # Save classification report
    clas_report_df.to_csv(r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\features\cnn_class_report.csv")
    print(clas_report_df)

def save_trained_model(model):
    # Save trained model
    model_path = r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\models\cnn_model.h5"
    model.save(model_path)
    print("Saved trained model at:", model_path)


if __name__ == "__main__":
    print("Training started")
    X = joblib.load(r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\features\X.joblib")
    y = joblib.load(r"C:\Users\HP\speech-emotion-recognition\speech_emotion_recognition\features\y.joblib")
    
    
    # print("CNN model is starting")
    # cnn_model(X, y)
    # print("Model finished.")
    print("LSTM model is starting")
    lstm_model(X, y)
    print("LSTM model is sucessfull")
    
    