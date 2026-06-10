import os
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import gradio as gr
from dotenv import load_dotenv

# 1. Load Environment Variables (.env file)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("ERROR: GROQ_API_KEY not found. Ensure your .env file is set up correctly.")

# 2. Connect to ChromaDB and local embedding function
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_collection(name="unofficial_guide", embedding_function=embed_fn)

# 3. Initialize Groq Client
groq_client = Groq(api_key=GROQ_API_KEY)

# 4. Define the RAG Query Engine
def answer_question(user_query):
    # Step A: Retrieve top 4 chunks matching the query from ChromaDB
    results = collection.query(
        query_texts=[user_query],
        n_results=4
    )
    
    # Extract retrieved text and sources
    retrieved_chunks = results['documents'][0]
    retrieved_metadatas = results['metadatas'][0]
    
    # Format the context block for the prompt
    context = "\n\n".join([f"[Source: {meta['source']}]\n{text}" for text, meta in zip(retrieved_chunks, retrieved_metadatas)])
    
    # Extract source filenames uniquely to display as citations
    sources = sorted(list(set([meta['source'] for meta in retrieved_metadatas])))
    source_citation = ", ".join(sources) if sources else "None found"
    
    # Step B: Construct the Grounded System Prompt (Milestone 5 Guardrail)
    system_prompt = (
        "You are a helpful assistant for university dorm reviews.\n"
        "Answer the user's question using ONLY the provided context blocks below.\n"
        "If the answer cannot be found or deduced from the context, explicitly reply exactly with:\n"
        "'I do not have enough information on that from the source documents.'\n"
        "Do not use external knowledge or make assumptions outside the provided text."
    )
    
    user_prompt = f"Context:\n{context}\n\nQuestion: {user_query}"
    
    # Step C: Send payload to Groq LLM
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0  # Keep temperature at 0 to guarantee grounding accuracy
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        answer = f"Error calling Groq API: {str(e)}"
        
    return answer, source_citation

# 5. Build and Launch the Gradio Web Interface
demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about the dorms (e.g., Which dorm has the best AC?)...", label="User Query"),
    outputs=[
        gr.Textbox(label="System Answer", interactive=False),
        gr.Textbox(label="Sources Consulted (Citations)", interactive=False)
    ],
    title="🏢 The Unofficial Campus Housing Guide",
    description="Ask real campus questions powered by community-sourced text files and grounded LLM generation.",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()