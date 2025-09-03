# config.py

# --- Model Configuration ---
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
LLM_MODEL = 'gemini-1.5-flash'
WHISPER_MODEL_CPU = 'base'  # Use 'tiny' or 'base' for faster performance on CPU
WHISPER_MODEL_GPU = 'small'  # Consider 'small' or 'medium' for better performance on GPU
SPACY_MODEL = 'en_core_web_sm'

# --- RAG Configuration ---
CHUNK_SIZE = 1000
NUM_RETRIEVAL_RESULTS = 5

# --- File & Database Paths ---
DB_PATH = "./chroma_db"
DB_COLLECTION_NAME = "video_transcript"
SUMMARY_JSON_FILE = "summary.json"
TRANSCRIPT_FILE = "transcript.txt"
AUDIO_FILE = "audio.mp3"

# --- API Keys ---
# The API key will be loaded from the .env file, but we can define the variable name here
GOOGLE_API_KEY_NAME = "GOOGLE_API_KEY"

# --- YouTube URL ---
YOUTUBE_URL = "https://youtu.be/SC2eSujzrUY?si=Rab4PBB6hUtU1eUI"