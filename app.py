from fastapi import FastAPI
from pydantic import BaseModel
import chromadb


app = FastAPI(title="PDF Vector Search API")


# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("pdf_chunks")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


@app.post("/query")
def search(request: QueryRequest):
    return [0.0]*384

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=request.top_k
    )

    response = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, distance in zip(documents, metadatas, distances):
        response.append({
            "chunk_text": doc,
            "page_number": meta["page_number"],
            "score": round(1 - distance, 4)
        })

    return response
