import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_transcript(audio_file_path):
    # Initialize the generative model
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Upload the audio file
    audio = genai.upload_file(path=audio_file_path)
    
    # Generate content with a prompt to extract the transcript
    response = model.generate_content(['extract well structured text from audio', audio])
    
    # Return the transcript text
    return response.text

# Example usage:
audio_file_path = '/Users/macbook/Desktop/project/extra/harvard.wav'
transcript = extract_transcript(audio_file_path)
print(transcript)


# summry with api--------------------------------------------------------------------------------------------------------------------

# import speech_recognition as sr
# from google.generativeai import GenerativeModel

# # Initialize the recognizer
# recognizer = sr.Recognizer()

# # Load the audio file
# audio_file_path = '/Users/macbook/Desktop/project/extra/harvard.wav'
# with sr.AudioFile(audio_file_path) as source:
#     audio_data = recognizer.record(source)

# # Transcribe the audio file
# try:
#     transcription = recognizer.recognize_google(audio_data)
#     print("Transcription: ", transcription)
    
#     # Initialize the generative model
#     model = GenerativeModel('gemini-pro')

#     # Generate content based on the transcribed text
#     response = model.generate_content([f'what is spoken in audio file: {transcription}'])

#     # Print the response
#     print(response.text)
# except sr.UnknownValueError:
#     print("Google Speech Recognition could not understand audio")
# except sr.RequestError as e:
#     print(f"Could not request results from Google Speech Recognition service; {e}")


# with summry without api-----------------------------------------------------------------------------------------------------

# import speech_recognition as sr
# from transformers import pipeline

# # Initialize the recognizer
# recognizer = sr.Recognizer()

# # Load the audio file
# audio_file_path = '/Users/macbook/Desktop/project/harvard.wav'
# with sr.AudioFile(audio_file_path) as source:
#     audio_data = recognizer.record(source)

# # Transcribe the audio file
# try:
#     transcription = recognizer.recognize_google(audio_data)
#     print("Transcription: ", transcription)
    
#     # Initialize the summarization pipeline
#     # summarizer = pipeline("summarization")

#     # Summarize the transcribed text
#     # summary = summarizer(transcription, max_length=150, min_length=30, do_sample=False)
    
#     # Print the summary
#     # print("Summary: ", summary[0]['summary_text'])
# except sr.UnknownValueError:
#     print("Google Speech Recognition could not understand audio")
# except sr.RequestError as e:
#     print(f"Could not request results from Google Speech Recognition service; {e}")








# video = genai.upload_file(path = '/Users/macbook/Desktop/project/harvard.wav')

# import time
# while video.state.name == 'processing':
#     print('.',end="")
    
#     time.sleep(10)
#     video = genai.get_file(video.name)
    
#     print('video uplded successfully')

# prompt = "provie brief summary of video in english"
# response = model.generate_content([prompt, video])
# print(response.text)
