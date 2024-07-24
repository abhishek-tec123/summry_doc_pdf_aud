from pydub import AudioSegment

# Load the MP3 file
mp3_file = "/Users/macbook/Desktop/project/extra/Let-Me-Love-You.mp3"  # Replace with your MP3 file path
audio = AudioSegment.from_mp3(mp3_file)

# Export as WAV
wav_file = "output_file.wav"  # Replace with desired WAV file path
audio.export(wav_file, format="wav")

print(f"Conversion complete: {wav_file}")
