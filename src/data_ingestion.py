import os
import subprocess
import time
import threading
import whisper
import torch
import ux.loading_anim_cli as loading
import config as config
import sys


def download_audio(video_url, output_filename=config.AUDIO_FILE):
    """Downloads the audio from a YouTube URL using yt-dlp."""
    output_path = os.path.splitext(output_filename)[0]
    command = [
        'yt-dlp',
        '-x', '--audio-format', 'mp3',
        '-o', f'{output_path}.%(ext)s',
        video_url
    ]
    print("Starting audio download...")
    subprocess.run(command, check=True)
    print(f"Audio downloaded successfully as '{output_path}.mp3'")
    return f'{output_path}.mp3'



def transcribe_audio(audio_path, output_filename=config.TRANSCRIPT_FILE):
    """Transcribes an audio file using Whisper and times the process."""
    # Detect if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    # Pick model size depending on device
    model_name = config.WHISPER_MODEL_GPU if device == "cuda" else config.WHISPER_MODEL_CPU # Use "small" for GPU, "base" for CPU (base here for faster demo)

    print("Loading Whisper model (base/small based on CPU/GPU availability)...")
    print(f"Loading Whisper model '{model_name}' on {device.upper()}...")
    
    model = whisper.load_model(model_name, device=device)

    # --- Spinner Setup ---
    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(target=loading.spinner_animation, args=(stop_spinner,))
    
    print("Starting transcription... (This may take a while) ", end="")
    spinner_thread.start()
    # ---------------------
    
    # --- Timer Start ---
    start_time = time.time()
    # -------------------
    
    try:
        # This is the long-running task
         result = model.transcribe(audio_path, fp16=(device == "cuda"))

    finally:
        # --- Stop the Spinner ---
        stop_spinner.set()
        spinner_thread.join()
        # ------------------------
    
    # Clear the spinner character and print a new line
    sys.stdout.write('\b \b')
    print("")
    
    # --- Timer End ---
    end_time = time.time()
    elapsed_time = end_time - start_time
    # -----------------
    
    # Format the time for readability (e.g., "1 minute(s) and 32 second(s)")
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"\nTranscription finished in {minutes} minute(s) and {seconds} second(s).")

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(str(result["text"]))
        
    print(f"Transcription saved to '{output_filename}'")
    return output_filename


def get_transcript(video_url, transcript_filename="transcript.txt"):
    """
    Gets the transcript for a video.
    Prioritizes human-created captions, falls back to Whisper if not available.
    """
    print("Attempting to download human-created English captions...")
    caption_command = [
        'yt-dlp',
        '--write-caption', '--sub-lang', 'en',
        '--skip-download', '-o', 'captions',
        video_url
    ]
    try:
        subprocess.run(caption_command, check=True, capture_output=True, text=True)
        caption_file = "captions.en.vtt"
        if os.path.exists(caption_file):
            print("Human-created captions found. Converting to plain text.")
            with open(caption_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            transcript_text = ""
            for line in lines:
                if "-->" not in line and "WEBVTT" not in line and line.strip():
                    transcript_text += line.strip() + " "
            with open(transcript_filename, 'w', encoding='utf-8') as f:
                f.write(transcript_text)
            os.remove(caption_file)
            return transcript_filename
        else:
            raise FileNotFoundError
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("No human-created captions found. Falling back to Whisper.")
        audio_file = download_audio(video_url)
        transcribed_file = transcribe_audio(audio_file)
        os.remove(audio_file)
        return transcribed_file