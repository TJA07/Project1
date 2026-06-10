# The Unofficial Guide — Project 1

---

## Domain
Michigan State Dorm reviews. These are scattered and hard to pinpoint one problem or find one place to detail all the good and bad. Promotional websites only list the good of MSU dorms, whilst reddit will detail everything. 
## Documents
1. dorm1.txt (Reddit review thread)
2. dorm2.txt (Reddit review thread)
3. dorm3.txt (Reddit review thread)
4. dorm4.txt (Reddit review thread)
5. dorm5.txt (Reddit review thread)
6. dorm6.txt (Reddit review thread)
7. dorm7.txt (Reddit review thread)
8. dorm8.txt (Reddit review thread)
9. dorm9.txt (Reddit review thread)
10. dorm10.txt (Reddit review thread)

## Chunking Strategy
I am using a fixed-size character chunking strategy set to 250 characters with a 50 character overlap. 

Our data consists of a distinct mix of both ultra-short, single-sentence student reviews and longer, multi-paragraph forum posts. A smaller chunk size of 250 characters ensures that short, punchy reviews do not get combined and diluted with unrelated feedback from other reviews. The 50-character overlap guarantees semantic continuity so that sentences spanning across a boundary do not lose their structural context during retrieval.

## Retrieval Approach
The embedding model used is `all-MiniLM-L6-v2` via the `sentence-transformers` library, with chunks indexed locally in ChromaDB. The pipeline will pull the top 4 most relevant chunks per user query ($k=4$). 

If deploying to real production users without cost constraints, I would evaluate an enterprise API model that has much greater power and ability/ Production tradeoffs to weigh include its support for vastly larger context windows and can understand slang and other topics significantly better. Running `all-MiniLM-L6-v2` locally offers zero runtime API costs and quick query latency, which perfectly serves this entry project to learn about AI agents.

## Evaluation Plan
1. Which dorm is noted by students as having the best AC and heating units?
2. What are the common student complaints regarding the noise levels in Wilson Hall?
3. Are the communal bathrooms in Brody Hall described as clean or dirty?
4. Which specific residence hall is located closest to the campus dining hall?
5. What are the primary structural complaints reported by students living in Wilson Hall?

## Anticipated Challenges
1. Text Fragmentation: Because 250 characters is a highly aggressive and small boundary, a multi-sentence review might get chopped poorly, requiring careful tuning of the 50-character overlap.
2. Slang: Language that is specifc to Michigan State, like spartan pride, or spartys market, will not be familiar and will struggle to get the context right. 

## AI Tool Plan
- **Database Engine Generation:** I plan to feed the AI my exact chunking strategy (250 character limit, 50 character overlap) and data directory configurations, requesting a script to ingest local text files directly into ChromaDB.
- **Frontend Assembly:** I will provide the AI with my target retrieval variables and ask it to output a clean Gradio interface layout featuring designated panels for text processing outputs and tracking source metadata.

## Architecture
```text
[Document Ingestion: Local .txt files] 
               │
               ▼
[Chunking: 250 Char Limit / 50 Overlap via Python] 
               │
               ▼
[Embedding & Indexing: all-MiniLM-L6-v2 -> ChromaDB] 
               │
               ▼
[Retrieval: Vector Semantic Search (k=4)] 
               │
               ▼
[Generation: Groq llama-3.3-70b-versatile Prompt System] 
               │
               ▼
[UI Layer: Gradio Web Interface Application]