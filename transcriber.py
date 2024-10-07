import argparse
import os
import subprocess
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

if check_ffmpeg():
    import moviepy.editor as mp
else:
    mp = None

def extract_audio(video_path, audio_path):
    if mp is None:
        raise RuntimeError("FFmpeg is not installed. Please install FFmpeg to use this feature.")
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

def preprocess_audio(audio_path):
    audio = AudioSegment.from_wav(audio_path)
    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_frame_rate(16000)  # Set frame rate to 16kHz
    audio.export(audio_path, format="wav")

def transcribe_audio_google(recognizer, audio):
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""

def transcribe_audio_sphinx(recognizer, audio):
    try:
        return recognizer.recognize_sphinx(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print(f"Sphinx error; {e}")
        return ""

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    
    # Load audio file
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    
    # Transcribe using multiple services
    google_transcript = transcribe_audio_google(recognizer, audio)
    sphinx_transcript = transcribe_audio_sphinx(recognizer, audio)
    
    # Return the longer transcript (assuming it's more accurate)
    return google_transcript if len(google_transcript) > len(sphinx_transcript) else sphinx_transcript

def process_large_audio(audio_path):
    print("Processing large audio file...")
    sound = AudioSegment.from_wav(audio_path)
    chunks = split_on_silence(sound, min_silence_len=1000, silence_thresh=sound.dBFS-14, keep_silence=500)
    
    folder_name = "audio-chunks"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    
    for i, audio_chunk in enumerate(chunks, start=1):
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        text = transcribe_audio(chunk_filename)
        print(f"Chunk {i}: {text}")
        whole_text += text + " "
    
    return whole_text

def process_video(video_path, output_path):
    audio_path = "temp_audio.wav"
    
    print(f"Extracting audio from {video_path}...")
    extract_audio(video_path, audio_path)
    
    print("Preprocessing audio...")
    preprocess_audio(audio_path)
    
    print("Starting transcription process...")
    transcription = process_large_audio(audio_path)
    
    with open(output_path, "w") as f:
        f.write(transcription)
    
    os.remove(audio_path)
    print(f"Transcription process completed. Output saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe MP4 videos to text.")
    parser.add_argument("input_file", help="/Users/sethegger/Downloads/scrum.mp4")
    parser.add_argument("output_file", help="transcription.txt")
    args = parser.parse_args()

    if not check_ffmpeg():
        print("Warning: FFmpeg is not installed. Audio extraction will not be available.")
        return

    try:
        process_video(args.input_file, args.output_file)
        print(f"Transcription completed successfully. Saved to {args.output_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()