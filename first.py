from flask import Flask, jsonify, request
from flask_cors import CORS
import pyaudio
import wave
import datetime
import speech_recognition as sr
from textblob import TextBlob
import threading
import os
import nltk
from transformers import pipeline
nltk.download('averaged_perceptron_tagger_eng')
from nltk.tag import PerceptronTagger
from nltk import pos_tag, word_tokenize
from transformers import pipeline
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

genai.configure(api_key="AIzaSyANmqxIG22VDP2lLV9FoynMy7_R5KQMJJ0")

# Initialize Tagger
tagger = PerceptronTagger(load=True)
print(pos_tag(word_tokenize("Artificial Intelligence is amazing")))

nltk.download('stopwords')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

audio = pyaudio.PyAudio()
stream = None
frames = []
is_recording = False
file_name = ""
recording_thread = None

# GPT Model for Future Prediction
future_predictor = pipeline("text-generation", model="gpt2")

# Start Recording
def capture_audio():
    global frames, stream, is_recording
    try:
        while is_recording:
            data = stream.read(1024)
            frames.append(data)
    except Exception as e:
        print(f"Error during recording: {e}")

@app.route('/start-recording', methods=['GET'])
def start_recording():
    global stream, frames, is_recording, file_name, recording_thread

    if is_recording:
        return jsonify({"error": "Recording is already in progress"}), 400

    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"recording_{timestamp}.wav"
        frames = []
        is_recording = True

        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        recording_thread = threading.Thread(target=capture_audio, daemon=True)
        recording_thread.start()

        print("Recording started...")
        return jsonify({"message": "Recording started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stop-recording', methods=['GET'])
def stop_recording():
    global stream, is_recording, recording_thread

    if not is_recording:
        return jsonify({"error": "No recording in progress"}), 400

    try:
        is_recording = False
        if recording_thread:
            recording_thread.join()

        if stream is not None:
            stream.stop_stream()
            stream.close()

        with wave.open(file_name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b"".join(frames))

        print(f"Audio saved as {file_name}")

        # Convert to text, analyze sentiment, extract main topic, and predict future
        transcription, sentiment, sentiment_score, main_topic, future_prediction = process_audio(file_name)
        return jsonify({
            "message": "Recording stopped",
            "transcription": transcription,
            "sentiment": sentiment,
            "sentiment_score": round(sentiment_score, 2),
            "main_topic": main_topic,
            "future_prediction": future_prediction
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Process Audio to Text, Perform Sentiment Analysis, Extract Main Topic, and Predict Future
def process_audio(file_name):
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(file_name) as source:
            print("Processing audio...")
            audio_data = recognizer.record(source)
            print("Converting to text...")
            text = recognizer.recognize_google(audio_data)
            print("Transcription: ", text)

            # Perform Sentiment Analysis with Updated Threshold
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity

            if sentiment_score > 0.25:
                sentiment = "Positive"
            elif sentiment_score < -0.25:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

            print(f"Sentiment: {sentiment} (Score: {sentiment_score:.2f})")

            # Save Transcription to File
            save_transcript(file_name, text)

            # Extract Main Topic and Save
            main_topic = extract_main_topic_bert(text)
            save_main_topic(file_name, main_topic)

            # Predict Future
            future_prediction = predict_future(text)
            print("Future Prediction: ", future_prediction)

            return text, sentiment, sentiment_score, main_topic, future_prediction
    except sr.UnknownValueError:
        return "Speech could not be understood", "Neutral", 0.0, "N/A", "N/A"
    except sr.RequestError as e:
        return f"Error: {e}", "Neutral", 0.0, "N/A", "N/A"
    
# Save Transcription to File
def save_transcript(file_name, text):
    try:
        if not os.path.exists("transcripts"):
            os.makedirs("transcripts")

        transcript_file = os.path.join("transcripts", os.path.basename(file_name).replace(".wav", ".txt"))
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Transcript saved to {transcript_file}")
    except Exception as e:
        print(f"Failed to save transcript: {e}")

# Save Main Topic to File
def save_main_topic(file_name, main_topic):
    try:
        if not os.path.exists("main_topics"):
            os.makedirs("main_topics")

        main_topic_file = os.path.join("main_topics", os.path.basename(file_name).replace(".wav", "_main_topic.txt"))
        with open(main_topic_file, "w", encoding="utf-8") as f:
            f.write(main_topic)
        print(f"Main topic saved to {main_topic_file}")
    except Exception as e:
        print(f"Failed to save main topic: {e}")

# Extract Main Topic Using BERT
def extract_main_topic_bert(text):
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        return "No sentences found in text."
    
    print("Extracting keywords using TF-IDF...")
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
    X = vectorizer.fit_transform(sentences)
    
    keywords = vectorizer.get_feature_names_out()
    
    if keywords.size > 0:
        print("Keywords Extracted:", keywords)
        return ', '.join(keywords)
    else:
        return "No clear topic detected"

# Predict Future Using GEMINI
def predict_future(text):
    try:
        prompt = (
            f"Based on this conversation: '{text}', predict the most likely outcomes, challenges, and opportunities. "
            "Provide actionable suggestions to improve the situation or maximize success in a clear, concise format. "
            "Answer short using the following structure: "
            "\n\n**Which Topic:** Identify the primary topic discussed. "
            "\n\n**Main Points:** Highlight the key points of the conversation. "
            "\n\n**Future of This Topic:** Predict the possible outcomes. "
            "\n\n**What Else Can Be Done With This Topic:** Suggest additional actions. "
            "\n\n**What We Should Not Do in the Future:** Provide warnings or actions to avoid."
        )
        model = genai.GenerativeModel('gemini-2.0-pro-exp')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return "Prediction unavailable"



if __name__ == "__main__":
    app.run(port=5001, debug=True)