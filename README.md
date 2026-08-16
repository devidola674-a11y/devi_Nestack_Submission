# Nestack PDF Vectorisation Pipeline

## Overview
This project implements a PDF vectorisation pipeline using Python.

### Technologies Used
- Python 3.12
- FastAPI
- ChromaDB
- Sentence Transformers (all-MiniLM-L6-v2)
- PyMuPDF

## Setup

Install dependencies:

bash
pip install -r requirements.txt


Run the ingestion script:

bash
python ingest.py --file "assessment_standard_vectorization (1).pdf"


Run the API:

bash
uvicorn app:app --reload


Open:

http://127.0.0.1:8000/docs

## Chunking Strategy

- Chunk Size: 500 characters
- Overlap: 100 characters

This preserves context while avoiding overly large chunks.

## Embedding Model

SentenceTransformer (all-MiniLM-L6-v2)

Reason:
- Fast
- Free
- High-quality semantic embeddings

## Vector Database

ChromaDB

Reason:
- Lightweight
- Easy local storage
- Fast similarity search

## Sample Query

POST /query

json
{
  "query": "What are the deliverables?",
  "top_k": 3
}


Deployment Link:

https://devi-nestack-submission-1.onrender.com
