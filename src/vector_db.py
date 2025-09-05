import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
import config as config



def create_vector_store(json_path, collection_name= config.DB_COLLECTION_NAME):
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
    text_chunks = [transcript[i:i + config.CHUNK_SIZE] for i in range(0, len(transcript), config.CHUNK_SIZE)]
    print(f"Transcript split into {len(text_chunks)} chunks.")
    
    # 3. Load the embedding model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 4. Create embeddings
    print("Creating embeddings for text chunks...")
    embeddings = model.encode(text_chunks, show_progress_bar=True)
    
    # 5. Initialize ChromaDB and create a collection
    # This will create a local folder to store the database.
    client = chromadb.PersistentClient(path=config.DB_PATH)

    # Check if the collection already exists and delete it
    if config.DB_COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"Deleting existing collection: '{config.DB_COLLECTION_NAME}'")
        client.delete_collection(name=config.DB_COLLECTION_NAME)

    # Create a new, empty collection
    collection = client.create_collection(name=config.DB_COLLECTION_NAME)
    # ----------------------
    collection = client.get_or_create_collection(name=config.DB_COLLECTION_NAME)
    
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