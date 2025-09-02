from sentence_transformers import SentenceTransformer
import chromadb
import os
from dotenv import load_dotenv
import google.generativeai as genai
import config as config
import ux.loading_anim_cli as loading
import data_ingestion as digest
import nlp_processor as nlp
from vector_db import create_vector_store

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
YOUR_API_KEY = os.getenv(config.GOOGLE_API_KEY_NAME)

if not YOUR_API_KEY:
    raise ValueError("Gemini API key not found. Please set the GOOGLE_API_KEY in your private-env file.")

genai.configure(api_key=YOUR_API_KEY) # type: ignore

def answer_question(question, collection_name="video_transcript"):
    """
    Answers a question based on the indexed transcript.
    """
    # 1. Initialize models and database
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    llm = genai.GenerativeModel(config.LLM_MODEL) # type: ignore
    client = chromadb.PersistentClient(config.DB_PATH)
    collection = client.get_collection(name=collection_name)
    
    # 2. Create an embedding for the user's question
    question_embedding = embedding_model.encode(question).tolist()
    
    # 3. Query the vector database for relevant chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results= config.NUM_RETRIEVAL_RESULTS  # Retrieve the top 5 most relevant chunks
    )
    context = "\n".join(results['documents'][0]) # type: ignore
    
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



if __name__ == '__main__':
    YOUTUBE_URL = config.YOUTUBE_URL 
    print(f"Processing video: {YOUTUBE_URL}")
    try:
        final_transcript_file = digest.get_transcript(YOUTUBE_URL)
        print(f"\n✅ Success! Transcript is ready in '{final_transcript_file}'")
        
        summary_file = nlp.enrich_and_save_json(final_transcript_file)

        # summary_file = "summary.json"
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