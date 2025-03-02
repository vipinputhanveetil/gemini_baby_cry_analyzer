from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
import librosa
import numpy as np
import google.generativeai as genai

genai.configure(api_key=os.getenv("GENAI_API_KEY"))


## Function to load OpenAI model and get respones
def get_gemini_response(prompt):
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    return response.text

## Function to extract audio features and generate prompt
def get_audio_features(audio_file):
    audio_path = audio_file
    y, sr = librosa.load(audio_path)

    # Extract audio features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCCs
    features=np.mean(mfccs, axis=1)  # Use mean values as features
    duration = librosa.get_duration(y=y, sr=sr)  # Duration of the audio
    pitch = librosa.piptrack(y=y, sr=sr)  # Pitch (fundamental frequency)
    intensity = np.abs(y)  # Intensity (amplitude of the audio signal)


    # Print analysis results
    print(f"Duration: {duration:.2f} seconds")
    print(f"Average Intensity: {np.mean(intensity):.4f}")
    print(f"Maximum Intensity: {np.max(intensity):.4f}")
    print(f"Average Pitch: {np.mean(pitch):.2f} Hz")
    print(f"Features: {features}")

    prompt_text = f"""
    **Prompt for AI Model:**
    I have analyzed my baby's crying audio using Python. Here are the results:
    - Duration: {duration:.2f} seconds
    - Average Intensity: {np.mean(intensity):.4f}
    - Maximum Intensity: {np.max(intensity):.4f}
    - Average Pitch: {np.mean(pitch):.2f} Hz
    - Features: {features}

    **Context:**
    The audio is of a baby crying, and I want to understand the reason for the crying based on the extracted features.
    The features include duration, intensity, pitch, and additional audio characteristics.

    **Task:**
    Analyze the provided features and suggest possible reasons for the baby's crying. Consider the following:
    1. Duration: How long the crying lasts.
    2. Intensity: How loud the crying is (higher values may indicate distress).
    3. Pitch: The tone of the crying (higher pitch may indicate discomfort or urgency).
    4. Features: Additional audio characteristics that may provide insights into the crying pattern.

    **Possible Reasons for Crying:**
    1. Hunger: Short, rhythmic cries with moderate intensity.
    2. Pain or Discomfort: High-pitched, intense, and prolonged cries.
    3. Sleepiness: Low-pitched, whiny cries with irregular patterns.
    4. Attention-Seeking: Intermittent cries with varying intensity and pitch.

    **Instructions for AI Model:**
    Based on the provided features, analyze the crying audio and suggest the most likely reason for the baby's crying. Provide a detailed explanation.
    """
    #print(prompt_text)
    return(prompt_text)

##initialize our streamlit app
st.set_page_config(page_title="Baby Cry Analyzer")
st.header("Baby Cry Analyzer")
uploaded_file = st.file_uploader("Upload an MP3 file", type=["mp3"])

## If file is uploaded
if uploaded_file is not None:
    # Display some information about the uploaded file
    st.write("File name:", uploaded_file.name)
    st.write("File type:", uploaded_file.type)
    st.write("File size:", uploaded_file.size, "bytes")
submit=st.button("Click submit to analyze the audio")
## If ask button is clicked
if submit: 
    prompt_text=get_audio_features(uploaded_file)
    response=get_gemini_response(prompt_text)
    st.subheader("The Response is")
    st.write(response)
