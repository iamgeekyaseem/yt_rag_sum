import  spacy
import json
from collections import Counter
import config as config


def enrich_and_save_json(transcript_path, output_filename=config.SUMMARY_JSON_FILE):
    """
    Processes a transcript to extract entities, counts them,
    and saves everything to a structured JSON file.
    """
    print("\nStarting NLP enrichment and JSON creation...")
    nlp = spacy.load(config.SPACY_MODEL)
    
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
