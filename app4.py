from flask import Flask, render_template, request, send_file
from flask_socketio import SocketIO, emit
import os
from werkzeug.utils import secure_filename
import nltk
from nltk.tokenize import sent_tokenize
import moviepy.editor as mp
import whisper
from pydub import AudioSegment
import zipfile
import io
import threading
import subprocess
import concurrent.futures
import time
import logging
import requests
from bs4 import BeautifulSoup
import yt_dlp
from yt_dlp.utils import DownloadError
import keyring
import getpass
from requests.auth import HTTPBasicAuth
from urllib.parse import urlparse, urljoin

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Ensure NLTK data is available
def check_nltk_data():
    nltk.data.path.append('/usr/share/nltk_data')
    logger.info(f"NLTK data path: {nltk.data.path}")
    try:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        logger.info("NLTK 'punkt' data loaded successfully.")
    except LookupError as e:
        logger.error(f"NLTK 'punkt' data not found. Error: {str(e)}")
        logger.error("NLTK data path contents:")
        for path in nltk.data.path:
            if os.path.exists(path):
                logger.error(f"Contents of {path}:")
                for root, dirs, files in os.walk(path):
                    for file in files:
                        logger.error(os.path.join(root, file))
            else:
                logger.error(f"Path does not exist: {path}")
        sys.exit(1)

check_nltk_data()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading')

# Ensure the upload and output folders exist
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'transcriptions'
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Set a timeout for each file processing (in seconds)
FILE_PROCESSING_TIMEOUT = 600  # 10 minutes

# Global variables
total_files = 0

# Confluence API settings
CONFLUENCE_URL = "https://your-confluence-instance.atlassian.net"
CONFLUENCE_USERNAME = "your_username"
CONFLUENCE_API_TOKEN = "your_api_token"

def get_credentials(service_name, force_prompt=False):
    username = keyring.get_password(service_name, "username")
    password = keyring.get_password(service_name, "password")
    
    if force_prompt or not username or not password:
        print(f"Please enter your {service_name} credentials:")
        username = input("Username: ")
        password = getpass.getpass("Password: ")
        store = input("Store credentials for future use? (y/n): ").lower() == 'y'
        if store:
            keyring.set_password(service_name, "username", username)
            keyring.set_password(service_name, "password", password)
    
    return username, password

def get_filename_from_url(url):
    path = urlparse(url).path
    filename = os.path.basename(path)
    return secure_filename(filename) or 'video'

def initial_cleanup():
    for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted file on startup: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    logger.info(f"Deleted directory on startup: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete {file_path}. Reason: {e}")

def extract_audio(video_path, audio_path):
    logger.debug(f"Extracting audio from {video_path}")
    try:
        command = [
            'ffmpeg',
            '-i', video_path,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            audio_path
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.debug("Audio extraction complete")
        logger.debug(f"ffmpeg stdout: {result.stdout}")
        logger.debug(f"ffmpeg stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr}")
        raise RuntimeError(f"FFmpeg error: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error in extract_audio: {str(e)}")
        raise

def preprocess_audio(audio_path):
    logger.debug(f"Preprocessing audio: {audio_path}")
    try:
        audio = AudioSegment.from_wav(audio_path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        audio.export(audio_path, format="wav")
        logger.debug("Audio preprocessing complete")
    except Exception as e:
        logger.error(f"Error preprocessing audio: {str(e)}")
        raise

# Load the model once
model = whisper.load_model("base")
model_lock = threading.Lock()

def transcribe_audio(audio_path):
    logger.debug(f"Transcribing audio: {audio_path}")
    with model_lock:
        result = model.transcribe(audio_path)
    logger.debug("Transcription complete")
    return result["text"]

def format_transcript(text):
    logger.debug("Formatting transcript")
    sentences = sent_tokenize(text)
    formatted_text = ""
    for i, sentence in enumerate(sentences):
        formatted_text += sentence + " "
        if (i + 1) % 3 == 0:
            formatted_text += "\n\n"
    logger.debug("Transcript formatting complete")
    return formatted_text.strip()

def process_video(video_path, total_files, current_file):
    logger.info(f"Starting to process video {current_file} of {total_files}: {video_path}")
    start_time = time.time()
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_name}_transcribed.txt")
    
    try:
        # Check if the file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"The file {video_path} does not exist.")

        # Ensure the file is in WAV format
        logger.info(f"Ensuring WAV format for file {current_file}")
        socketio.emit('update_progress', {'file': current_file, 'progress': 10, 'status': f'Preparing audio for file {current_file}'})
        wav_path = ensure_wav_format(video_path)
        logger.info(f"WAV format ensured for file {current_file}. Time taken: {time.time() - start_time:.2f} seconds")

        logger.info(f"Preprocessing audio for file {current_file}")
        socketio.emit('update_progress', {'file': current_file, 'progress': 30, 'status': f'Preprocessing audio for file {current_file}'})
        preprocess_audio(wav_path)
        logger.info(f"Audio preprocessing complete for file {current_file}. Time taken: {time.time() - start_time:.2f} seconds")
        
        logger.info(f"Transcribing audio for file {current_file}")
        socketio.emit('update_progress', {'file': current_file, 'progress': 60, 'status': f'Transcribing audio for file {current_file}'})
        transcription = transcribe_audio(wav_path)
        logger.info(f"Transcription complete for file {current_file}. Time taken: {time.time() - start_time:.2f} seconds")
        
        logger.info(f"Formatting transcript for file {current_file}")
        formatted_transcription = format_transcript(transcription)
        logger.info(f"Transcript formatting complete for file {current_file}. Time taken: {time.time() - start_time:.2f} seconds")
        
        with open(output_path, 'w') as f:
            f.write(formatted_transcription)
        
        logger.info(f"Removing temporary files for file {current_file}")
        os.remove(wav_path)  # Remove the processed WAV file
        
        socketio.emit('update_progress', {'file': current_file, 'progress': 95, 'status': f'Finished processing file {current_file}'})
        logger.info(f"Finished processing file {current_file}. Total time taken: {time.time() - start_time:.2f} seconds")
        return output_path
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        socketio.emit('update_progress', {'file': current_file, 'progress': 95, 'status': f'Error: File not found - {str(e)}'})
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}")
        socketio.emit('update_progress', {'file': current_file, 'progress': 95, 'status': f'Error: {str(e)}'})
    return None

def process_file_with_timeout(file_path, total_files, current_file):
    try:
        return process_video(file_path, total_files, current_file)
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return None

def process_files(file_paths):
    global total_files
    total_files = len(file_paths)
    logger.info(f"Starting to process {total_files} files")
    
    if total_files == 0:
        logger.warning("No files to process")
        socketio.emit('error', {'message': "No files were uploaded"})
        return

    output_files = []
    overall_timeout = 1800  # 30 minutes total timeout

    def process_all_files():
        nonlocal output_files
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_file = {executor.submit(process_file_with_timeout, file_path, total_files, i+1): (i+1, file_path) for i, file_path in enumerate(file_paths)}
            for future in concurrent.futures.as_completed(future_to_file):
                file_number, file_path = future_to_file[future]
                try:
                    output_path = future.result(timeout=FILE_PROCESSING_TIMEOUT)
                    if output_path:
                        output_files.append(output_path)
                        logger.info(f"Successfully processed file {file_number}: {file_path}")
                    else:
                        logger.warning(f"File {file_number} was processed but no output was generated: {file_path}")
                except concurrent.futures.TimeoutError:
                    logger.error(f"Processing of file {file_number} ({file_path}) timed out")
                    socketio.emit('update_progress', {'file': file_number, 'progress': 95, 'status': f'Error: Processing timed out for file {file_number}'})
                except Exception as e:
                    logger.error(f"Error processing file {file_number} ({file_path}): {str(e)}")
                    socketio.emit('update_progress', {'file': file_number, 'progress': 95, 'status': f'Error processing file {file_number}: {str(e)}'})

    process_thread = threading.Thread(target=process_all_files)
    process_thread.start()
    process_thread.join(timeout=overall_timeout)

    if process_thread.is_alive():
        logger.error("Overall processing timed out")
        socketio.emit('error', {'message': "Processing timed out. Please try again with fewer or smaller files."})
        return

    logger.info(f"Finished processing. {len(output_files)} out of {total_files} files were successful")
    
    socketio.emit('update_progress', {'file': 'all', 'progress': 98, 'status': 'Preparing files for download'})
    
    if len(output_files) == 1:
        logger.info(f"Emitting transcription complete for single file: {output_files[0]}")
        socketio.emit('transcription_complete', {'download_url': f'/download/{os.path.basename(output_files[0])}'})
    elif len(output_files) > 1:
        logger.info(f"Creating zip file for {len(output_files)} transcriptions")
        zip_filename = 'transcriptions.zip'
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            for f in output_files:
                zf.write(f, os.path.basename(f))
        memory_file.seek(0)
        
        zip_path = os.path.join(app.config['OUTPUT_FOLDER'], zip_filename)
        with open(zip_path, 'wb') as f:
            f.write(memory_file.getvalue())
        
        logger.info(f"Emitting transcription complete for zip file: {zip_filename}")
        socketio.emit('transcription_complete', {'download_url': f'/download/{zip_filename}'})
    else:
        logger.warning("No files were successfully processed")
        socketio.emit('error', {'message': "No files were successfully processed"})
    
    socketio.emit('update_progress', {'file': 'all', 'progress': 100, 'status': 'All files processed and ready for download'})
    socketio.emit('all_files_complete')

def download_authenticated_video(url, output_folder):
    max_attempts = 3
    attempt = 0
    
    filename = get_filename_from_url(url)
    output_path = os.path.join(output_folder, filename)
    
    while attempt < max_attempts:
        server_credentials = get_credentials("video_server", force_prompt=(attempt > 0))
        
        try:
            response = requests.get(url, auth=HTTPBasicAuth(server_credentials[0], server_credentials[1]), stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Successfully downloaded video from {url}")
            return output_path
        
        except requests.HTTPError as e:
            if e.response.status_code == 401:
                logger.warning("Authentication failed. Please try again.")
                attempt += 1
                if attempt == max_attempts:
                    raise ValueError("Maximum authentication attempts reached. Please check your credentials.")
            else:
                raise ValueError(f"HTTP Error occurred: {e}")
        
        except requests.RequestException as e:
            logger.error(f"Error downloading video from server: {str(e)}")
            raise ValueError(f"Failed to download video: {str(e)}")

def scrape_confluence_page(page_url):
    confluence_credentials = get_credentials("confluence")
    server_credentials = get_credentials("video_server")
    
    # Parse the Confluence URL
    parsed_url = urlparse(page_url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    page_id = parsed_url.path.split('/')[-1]
    
    # Initialize Confluence client
    confluence = Confluence(
        url=base_url,
        username=confluence_credentials[0],
        password=confluence_credentials[1]
    )
    
    try:
        # Get page content
        page_content = confluence.get_page_by_id(page_id, expand='body.storage')
        html_content = page_content['body']['storage']['value']
        
        # Parse HTML content
        soup = BeautifulSoup(html_content, 'html.parser')
        
        video_urls = []
        for iframe in soup.find_all('iframe'):
            src = iframe.get('src')
            if src:
                # Construct full URL if it's a relative path
                full_url = urljoin(base_url, src)
                video_urls.append(full_url)
        
        return video_urls
    except Exception as e:
        logger.error(f"Error scraping Confluence page: {str(e)}")
        raise

def download_video(url, output_folder):
    ydl_opts = {
        'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s'),
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'keepvideo': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            wav_filename = os.path.splitext(filename)[0] + '.wav'
            return wav_filename if os.path.exists(wav_filename) else filename
    except yt_dlp.utils.DownloadError as e:
        logger.error(f"Error downloading video: {str(e)}")
        raise ValueError(f"An error occurred while downloading the video: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in download_video: {str(e)}")
        raise ValueError(f"An unexpected error occurred: {str(e)}")

def ensure_wav_format(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        wav_path = os.path.splitext(file_path)[0] + '.wav'
        audio.export(wav_path, format='wav')
        if file_path != wav_path:
            os.remove(file_path)  # Remove the original file if it's not the same as the new WAV file
        return wav_path
    except Exception as e:
        logger.error(f"Error converting to WAV: {str(e)}")
        raise



@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file_paths = []
        
        # Handle file uploads
        if 'file' in request.files:
            files = request.files.getlist('file')
            for file in files:
                if file and file.filename != '':
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    file_paths.append(file_path)
        
        # Handle URL input
        url = request.form.get('url')
        if url:
            try:
                if 'confluence' in url.lower():
                    video_urls = scrape_confluence_page(url)
                    for video_url in video_urls:
                        if 'youtube.com' in video_url or 'vimeo.com' in video_url:
                            downloaded_path = download_video(video_url, app.config['UPLOAD_FOLDER'])
                        else:
                            downloaded_path = download_authenticated_video(video_url, app.config['UPLOAD_FOLDER'])
                        if downloaded_path:
                            file_paths.append(downloaded_path)
                else:
                    # Single video URL
                    if 'youtube.com' in url or 'vimeo.com' in url:
                        downloaded_path = download_video(url, app.config['UPLOAD_FOLDER'])
                    else:
                        downloaded_path = download_authenticated_video(url, app.config['UPLOAD_FOLDER'])
                    if downloaded_path:
                        file_paths.append(downloaded_path)
            except ValueError as e:
                return render_template('index.html', error=str(e))
            except Exception as e:
                logger.error(f"Error processing URL {url}: {str(e)}")
                return render_template('index.html', error="An unexpected error occurred while processing the URL.")
        
        if not file_paths:
            return render_template('index.html', error="No files were successfully uploaded or downloaded.")
        
        thread = threading.Thread(target=process_files, args=(file_paths,))
        thread.start()
        
        return render_template('progress.html')
    
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), as_attachment=True)

@socketio.on('get_total_files')
def handle_get_total_files():
    global total_files
    emit('set_total_files', {'total_files': total_files})

@socketio.on('connect')
def test_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def test_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on_error_default
def default_error_handler(e):
    logger.error(f"SocketIO error: {str(e)}")

if __name__ == '__main__':
    initial_cleanup()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)        