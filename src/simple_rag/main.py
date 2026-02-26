"""
Simple RAG (Retrieval-Augmented Generation) — CLI interface.

Usage:
    pixi run python -m simple_rag.main [path/to/document.pdf]
"""

import sys

from simple_rag.rag import ingest_pdf_path, query_document


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else input("PDF path: ").strip()

    print(f"Loading PDF: {pdf_path}")
    doc_id = ingest_pdf_path(pdf_path)
    print("Ready! Ask questions (type 'quit' to exit)\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        result = query_document(doc_id, query)
        print(f"\nAssistant: {result['answer']}\n")


if __name__ == "__main__":
    main()
