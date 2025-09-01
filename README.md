# Video Q\&A: Your Personal YouTube Researcher

This project is a complete, end-to-end pipeline that transforms a YouTube video into an interactive, searchable knowledge base. Ever wish you could just ask a question to a long lecture or news report and get a straight answer? This tool makes that possible. It ingests a video, processes its content, and allows you to have a conversation with it, getting answers based directly on what was said.

## Key Features

  * **Intelligent Transcription**: Automatically pulls high-quality, human-created captions if they exist. If not, it falls back to generating a highly accurate transcript using OpenAI's Whisper model.
  * **NLP Enrichment**: The raw transcript is processed to extract structured data, including Named Entity Recognition (NER) to identify people, organizations, and locations mentioned.
  * **Vector Indexing**: The transcript is chunked and converted into vector embeddings, which are stored in a local ChromaDB vector store for efficient semantic search.
  * **Interactive Q\&A**: Utilizes a Retrieval-Augmented Generation (RAG) architecture. When you ask a question, the system retrieves the most relevant text chunks and uses Google's Gemini LLM to generate a factual, source-based answer.
  * **Optimized Performance**: Features automatic GPU detection with PyTorch to accelerate transcription and an efficient chatbot class that loads models only once for rapid, successive questioning.

## How It Works: The Pipeline

The project operates through a sequential data processing pipeline, turning unstructured video content into a queryable database.

1.  **Ingestion**: The process starts with a YouTube URL. The script first checks for human-made English captions. If found, it uses them as the high-quality source. If not, it uses `yt-dlp` to download the audio stream.
2.  **Transcription**: If no captions were available, the downloaded audio is processed by a local Whisper model to generate an accurate text transcript. This stage includes a live timer to provide user feedback.
3.  **Enrichment**: The raw transcript is analyzed using spaCy to identify and count named entities. This structured information is then saved alongside the full text into a `summary.json` file, creating a reusable data asset.
4.  **Indexing**: The full transcript is divided into smaller text chunks. Each chunk is passed through a `SentenceTransformer` model to create a semantic vector embedding. These embeddings, along with the original text, are stored in a local ChromaDB vector database.
5.  **Querying**: When a user asks a question, the query is converted into an embedding. This embedding is used to find the most relevant text chunks from the vector database. These chunks, along with the original question, are fed to the Gemini LLM in a carefully crafted prompt, which generates the final answer.

## Tech Stack

  * **Core AI/ML**: OpenAI Whisper, spaCy, Sentence-Transformers, Google Gemini API, PyTorch
  * **Data & Ops**: ChromaDB, yt-dlp, python-dotenv
  * **Core Libraries**: `subprocess`, `threading`, `json`, `itertools`

## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

  * Python 3.10+
  * Conda for environment management
  * FFmpeg: A system-level dependency required for audio processing. On Windows, it's recommended to install it with Chocolatey (`choco install ffmpeg`).

### Installation & Setup

1.  **Clone the Repository**

    ```bash
    git clone https://your-repository-url.git
    cd your-repository-folder
    ```

2.  **Create and Activate Conda Environment**

    ```bash
    conda create --name aienv python=3.10
    conda activate aienv
    ```

3.  **Install Dependencies**
    First, create a `requirements.txt` file from your project by running `pip freeze > requirements.txt`. A new user can then install all necessary Python packages with:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download spaCy Model**
    You need the English language model for NLP enrichment.

    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Set Up API Key**

      * Create a file named `.env` in the root directory of the project.
      * Add your Google Gemini API key to this file:
        ```
        GOOGLE_API_KEY="your_api_key_here"
        ```
      * The `.env` file is included in `.gitignore` to keep your key secure.

## Usage

The script is designed to be run from the command line.

1.  **Full Pipeline Run (for a new video)**

      * In `main.py`, uncomment the lines in the `if __name__ == '__main__':` block that define the `YOUTUBE_URL` and call the processing functions (`get_transcript`, `enrich_and_save_json`, `create_vector_store`).
      * Run the script:
        ```bash
        python main.py
        ```
      * This will process the video from start to finish and then launch the interactive Q\&A session.

2.  **Interactive Q\&A (if data is already processed)**

      * Ensure the `chroma_db` folder exists from a previous run.
      * Make sure the initial processing lines in the `main` block are commented out.
      * Run the script to start the chatbot immediately:
        ```bash
        python main.py
        ```
      * Ask questions in the terminal and type `quit` to exit.

## Future Directions

This project serves as a strong foundation. Potential future improvements include:

  * **Web Interface**: Building a simple UI with Streamlit or Gradio to make the tool more accessible.
  * **Speaker Diarization**: Enhancing the transcript to identify who is speaking and when.
  * **Multi-Video Support**: Expanding the system to index and query an entire YouTube channel or playlist.
  * **REST API**: Wrapping the Q\&A functionality in a FastAPI endpoint to serve it as a web service.