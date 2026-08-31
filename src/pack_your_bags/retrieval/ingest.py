from pathlib import Path

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pack_your_bags.config import settings
from pack_your_bags.retrieval.embeddings import embed_texts
from pack_your_bags.retrieval.vector_store import clear_namespace, upsert_chunks

RULES_DIR = Path(__file__).resolve().parents[3] / "data" / "rules"

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    _, fm_block, body = raw.split("---", 2)
    return yaml.safe_load(fm_block), body.strip()


def load_documents() -> list[dict]:
    docs = []
    for path in sorted(RULES_DIR.glob("*.md")):
        frontmatter, body = parse_frontmatter(path.read_text())
        docs.append({"id": path.stem, "frontmatter": frontmatter, "body": body})
    return docs


def build_header(fm: dict) -> str:
    nationalities = ", ".join(fm.get("applicable_nationalities", []))
    return (
        f"Topic: {fm.get('topic', '')}\n"
        f"Destination: {fm.get('destination_country', '')}\n"
        f"Applicable nationalities: {nationalities}\n\n"
    )


def chunk_document(doc: dict) -> list[dict]:
    fm = doc["frontmatter"]
    header = build_header(fm)
    pieces = _splitter.split_text(doc["body"])

    chunks = []
    for i, piece in enumerate(pieces):
        text = header + piece
        metadata = {
            "source_url": fm.get("source_url", ""),
            "source_name": fm.get("source_name", ""),
            "fetched_on": str(fm.get("fetched_on", "")),
            "destination_country": fm.get("destination_country", ""),
            "applicable_nationalities": fm.get("applicable_nationalities", []),
            "topic": fm.get("topic", ""),
            "confidence": fm.get("confidence", ""),
            "chunk_index": i,
            "total_chunks": len(pieces),
            "text": text,
        }
        if "archived_via" in fm:
            metadata["archived_via"] = fm["archived_via"]
        if "superseded_note" in fm:
            metadata["superseded_note"] = fm["superseded_note"]

        chunk_id = doc["id"] if len(pieces) == 1 else f"{doc['id']}__chunk{i}"
        chunks.append({"id": chunk_id, "text": text, "metadata": metadata})
    return chunks


def run() -> None:
    docs = load_documents()
    print(f"Loaded {len(docs)} documents from {RULES_DIR}")

    chunks_per_doc = [chunk_document(doc) for doc in docs]
    all_chunks = [chunk for doc_chunks in chunks_per_doc for chunk in doc_chunks]
    split_docs = sum(1 for doc_chunks in chunks_per_doc if len(doc_chunks) > 1)
    print(f"Split into {len(all_chunks)} chunks ({split_docs} documents were split, "
          f"{len(docs) - split_docs} stayed as a single chunk)")

    embeddings = embed_texts([c["text"] for c in all_chunks])
    print(f"Embedded {len(embeddings)} chunks")

    vectors = [
        {"id": chunk["id"], "values": embedding, "metadata": chunk["metadata"]}
        for chunk, embedding in zip(all_chunks, embeddings)
    ]

    clear_namespace()
    print("Cleared existing vectors in namespace before re-upserting")

    upsert_chunks(vectors)
    print(f"Upserted {len(vectors)} vectors into Pinecone")


if __name__ == "__main__":
    run()
