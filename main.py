import subprocess
import os
import whisper # For transcription
import time
import sys
import itertools
import threading
import spacy
import json
from collections import Counter
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
YOUR_API_KEY = os.getenv("GOOGLE_API_KEY")

if not YOUR_API_KEY:
    raise ValueError("Gemini API key not found. Please set the GEMINI_API_KEY in your .env file.")

genai.configure(api_key=YOUR_API_KEY)

def answer_question(question, collection_name="video_transcript"):
    """
    Answers a question based on the indexed transcript.
    """
    # 1. Initialize models and database
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    llm = genai.GenerativeModel('gemini-1.5-flash')
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name=collection_name)
    
    # 2. Create an embedding for the user's question
    question_embedding = embedding_model.encode(question).tolist()
    
    # 3. Query the vector database for relevant chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=5  # Retrieve the top 5 most relevant chunks
    )
    context = "\n".join(results['documents'][0])
    
    # 4. Construct the prompt for the LLM 📄
    prompt_template = f"""
    You are a helpful assistant who answers questions based on the provided video transcript.
    Answer the following question based ONLY on the context below.
    If the answer is not in the context, say "I don't have enough information from the video to answer."

    CONTEXT:
    {context}

    QUESTION:
    {question}
    """
    
    # 5. Get the answer from the LLM
    response = llm.generate_content(prompt_template)
    
    return response.text


def download_audio(video_url, output_filename="audio.mp3"):
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

def spinner_animation(stop_event):
    """Displays a simple spinner animation in the console."""
    spinner = itertools.cycle(['|', '/', '-', '\\'])
    while not stop_event.is_set():
        sys.stdout.write(next(spinner))  # Write the character
        sys.stdout.flush()               # Flush the output
        sys.stdout.write('\b')           # Move the cursor back
        time.sleep(0.1)



def transcribe_audio(audio_path, output_filename="transcript.txt"):
    """Transcribes an audio file using Whisper and times the process."""
    # Detect if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    # Pick model size depending on device
    model_name = "base" if device == "cuda" else "base" # Use "small" for GPU, "base" for CPU (base here for faster demo)

    print("Loading Whisper model (base/small based on CPU/GPU availability)...")
    print(f"Loading Whisper model '{model_name}' on {device.upper()}...")
    
    model = whisper.load_model(model_name, device=device)

    # --- Spinner Setup ---
    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(target=spinner_animation, args=(stop_spinner,))
    
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

# def transcribe_audio(audio_path, output_filename="transcript.txt"):
#     """Transcribes an audio file using Whisper."""
#     print("Loading Whisper model (base)...")
#     model = whisper.load_model("base")
#     print("Starting transcription... (This may take a while)")
#     # Set fp16=False for wider compatibility (CPU-only).
#     result = model.transcribe(audio_path, fp16=False)
    
#     with open(output_filename, 'w', encoding='utf-8') as f:
#         f.write(str(result["text"]))
        
#     print(f"Transcription saved to '{output_filename}'")
#     return output_filename

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
    

def enrich_and_save_json(transcript_path, output_filename="summary.json"):
    """
    Processes a transcript to extract entities, counts them,
    and saves everything to a structured JSON file.
    """
    print("\nStarting NLP enrichment and JSON creation...")
    nlp = spacy.load("en_core_web_sm")
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript_text = f.read()
        
    doc = nlp(transcript_text)
    
    # 1. Extract unique entities (text and label)
    unique_entities = sorted(list(set([(ent.text.strip(), ent.label_) for ent in doc.ents])))
    
    # 2. Count the frequency of each entity text
    entity_texts = [ent.text.strip() for ent in doc.ents]
    entity_counts = Counter(entity_texts)
    
    # 3. Assemble the final data structure
    output_data = {
        "full_transcript": transcript_text,
        "summary_points": [], # We'll populate this in a future step
        "entities": [
            {"text": text, "label": label, "count": entity_counts[text]}
            for text, label in unique_entities
        ]
    }
    
    # 4. Write the data to a JSON file
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    print(f"Enriched data saved to '{output_filename}'")
    return output_filename


def create_vector_store(json_path, collection_name="video_transcript"):
    """
    Creates a vector store from the transcript in the JSON file.
    """
    print("\nCreating vector store for Q&A...")
    
    # 1. Load the transcript from the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    transcript = data['full_transcript']
    
    # 2. Chunk the transcript
    # A simple strategy: split the text into chunks of a certain size.
    # More advanced strategies could use sentence splitting (with spaCy or NLTK).
    text_chunks = [transcript[i:i + 1000] for i in range(0, len(transcript), 1000)]
    print(f"Transcript split into {len(text_chunks)} chunks.")
    
    # 3. Load the embedding model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 4. Create embeddings
    print("Creating embeddings for text chunks...")
    embeddings = model.encode(text_chunks, show_progress_bar=True)
    
    # 5. Initialize ChromaDB and create a collection
    # This will create a local folder to store the database.
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name=collection_name)
    
    # 6. Store the chunks and their embeddings in the collection
    print("Adding embeddings to the vector store...")
    for i, chunk in enumerate(text_chunks):
        collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embeddings[i].tolist()]
        )
        
    print(f"✅ Vector store created successfully with {collection.count()} items.")
    return collection


if __name__ == '__main__':
    # YOUTUBE_URL = "https://youtu.be/CxVXvFOPIyQ?si=QyziI5rKLPdcPF-s"
    # print(f"Processing video: {YOUTUBE_URL}")
    try:
    #     final_transcript_file = get_transcript(YOUTUBE_URL)
    #     print(f"\n✅ Success! Transcript is ready in '{final_transcript_file}'")
        
    #     summary_file = enrich_and_save_json(final_transcript_file)

        summary_file = "summary.json"
        print(f"✅ Enriched data saved to '{summary_file}'")
        
        # Add the new vector store creation step
        create_vector_store(summary_file)

        # --- Interactive Q&A Loop ---
        print("\n---")
        print("✅ Setup complete! You can now ask questions about the video.")
        print("Type 'quit' to exit.")
        
        while True:
            user_question = input("\nYour Question: ")
            if user_question.lower() == 'quit':
                break
                
            print("\nThinking...")
            answer = answer_question(user_question)
            print(f"\nAnswer: {answer}")

    except Exception as e:
        print(f"\n❌ An error occurred in the main process: {e}")