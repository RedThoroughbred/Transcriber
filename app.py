from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import nltk
from nltk.tokenize import sent_tokenize
import moviepy.editor as mp
import whisper
from pydub import AudioSegment

# Set up NLTK data path
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download necessary NLTK data
nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)

app = Flask(__name__)

# Ensure the upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_audio(video_path, audio_path):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

def preprocess_audio(audio_path):
    audio = AudioSegment.from_wav(audio_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export(audio_path, format="wav")

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

def format_transcript(text):
    sentences = sent_tokenize(text)
    formatted_text = ""
    for i, sentence in enumerate(sentences):
        formatted_text += sentence + " "
        if (i + 1) % 3 == 0:
            formatted_text += "\n\n"
    return formatted_text.strip()

def process_video(video_path):
    audio_path = "temp_audio.wav"
    extract_audio(video_path, audio_path)
    preprocess_audio(audio_path)
    transcription = transcribe_audio(audio_path)
    formatted_transcription = format_transcript(transcription)
    os.remove(audio_path)
    return formatted_transcription

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            transcription = process_video(file_path)

            # Delete the uploaded file
            try:
                os.remove(file_path) 
            except Exception as e:
                print(f"Error deleting file: {e}") 

            return render_template('transcript.html', transcription=transcription)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)