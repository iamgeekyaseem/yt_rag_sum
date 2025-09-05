# inspect_db.py
import chromadb
import config

# This script inspects the contents of the vector database and prints out all stored items.for testing purposes. helped me fix a bug in the main script.

# Connect to the persistent database
client = chromadb.PersistentClient(path=config.DB_PATH)

try:
    # Get the existing collection
    collection = client.get_collection(name=config.DB_COLLECTION_NAME)

    # Retrieve all items from the collection
    # ChromaDB's get() method can retrieve up to all items if no IDs are specified.
    # For very large collections, you might want to specify a limit.
    items = collection.get() 

    print(f"Found {len(items['ids'])} items in the '{config.DB_COLLECTION_NAME}' collection.\n")

    # Loop through and print each item
    for i, doc_id in enumerate(items['ids']):
        document = items['documents'][i]
        print(f"--- Item ID: {doc_id} ---")
        print(document)
        print("-" * (len(doc_id) + 15) + "\n")

except Exception as e:
    print(f"An error occurred: {e}")
    print("The collection might not exist yet. Run the main processing script first.")