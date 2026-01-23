# get_exact_chunk.py
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

host = os.getenv('QDRANT_URL') or os.getenv('QDRANT_HOST')
port = os.getenv('QDRANT_PORT', '6333')
api_key = os.getenv('QDRANT_API_KEY')
url = f"{host}:{port}" if ':' not in host.split('//')[-1] else host

client = QdrantClient(url=url, api_key=api_key)
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# Get first few chunks (no filter)
results = client.scroll(
    collection_name="bio_fulltext_chunks",
    limit=3,
    with_payload=True,
    with_vectors=True
)

print("=== FIRST 3 STORED CHUNKS ===")
for i, point in enumerate(results[0], 1):
    text = point.payload.get('text', '')[:100]
    doi = point.payload.get('doi', 'unknown')
    print(f"\n{i}. DOI: {doi}")
    print(f"   Text: {text}...")

# Test with EXACT stored text
if results[0]:
    exact_text = results[0][0].payload.get('text', '')
    stored_vector = results[0][0].vector
    
    print(f"\n\n=== TESTING VECTOR CONSISTENCY ===")
    print(f"Exact text: {exact_text[:80]}...")
    
    # Re-encode the exact text
    new_vector = model.encode(exact_text).tolist()
    
    # Compare vectors (cosine similarity)
    similarity = np.dot(stored_vector, new_vector) / (np.linalg.norm(stored_vector) * np.linalg.norm(new_vector))
    print(f"\n✅ Similarity between stored & re-encoded: {similarity:.4f}")
    
    if similarity > 0.99:
        print("   GOOD - Vectors are consistent!")
    else:
        print("   ⚠️  WARNING - Vectors don't match! Different model or settings used.")