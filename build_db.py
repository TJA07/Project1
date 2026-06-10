import os
import chromadb
from chromadb.utils import embedding_functions

# 1. Setup ChromaDB and Embedding Model (all-MiniLM-L6-v2 runs locally)
print("Initializing database and model...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Clear old database if you ran it previously, then create new
try:
    chroma_client.delete_collection("unofficial_guide")
except:
    pass
collection = chroma_client.create_collection(name="unofficial_guide", embedding_function=embed_fn)

# 2. Chunking Settings (Matching your spec exactly)
CHUNK_SIZE = 500
OVERLAP = 100

def get_chunks(text):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        # Move forward by the chunk size minus the overlap
        start += CHUNK_SIZE - OVERLAP 
    return chunks

# 3. Process Text Documents
doc_dir = "data"
total_chunks = 0

print(f"Reading files from {doc_dir}...")
for filename in os.listdir(doc_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(doc_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            
            # Clean empty text
            if not text.strip(): continue
            
            chunks = get_chunks(text)
            
            # 4. Load into Vector Store
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 10: 
                    collection.add(
                        documents=[chunk],
                        metadatas=[{"source": filename}],
                        ids=[f"{filename}_chunk_{i}"]
                    )
                    total_chunks += 1

print(f"Database built successfully! Loaded {total_chunks} total chunks.")