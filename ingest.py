import os
import argparse
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="pdf_chunks")


def extract_text(pdf_path):
    document = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text = page.get_text("text")
        pages.append({
            "page_number": page_num + 1,
            "text": text
        })

    return pages


def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def ingest(pdf_path):
    pages = extract_text(pdf_path)

    chunk_id = 0

    for page in pages:
        chunks = chunk_text(page["text"])

        for chunk in chunks:

            embedding = model.encode(chunk).tolist()

            collection.add(
                ids=[str(chunk_id)],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[
                    {
                        "page_number": page["page_number"]
                    }
                ]
            )

            chunk_id += 1

    print("\nPDF ingestion completed successfully.")
    print(f"Total chunks stored: {chunk_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        required=True,
        help="Path to PDF file"
    )

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print("PDF file not found.")
    else:
        ingest(args.file)