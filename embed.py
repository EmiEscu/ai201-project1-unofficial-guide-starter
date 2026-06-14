"""
embed.py — Phase 1 (tail) + Phase 2 (head): Embedding + Vector Store + Retrieval

The Unofficial Guide: UIC Course & Professor Reviews (RAG)

This script implements two stages of the architecture diagram in planning.md:

    Phase 1: Document Pipeline (ETL)
        Document Ingestion (raw .txt)
            -> Chunking            (chunking.py  -> chunks.json)
            -> Embedding           (all-MiniLM-L6-v2 via sentence-transformers)  <- here
            -> Vector Store        (ChromaDB, persisted to ./chroma_db)          <- here

    Phase 2: Query Pipeline (RAG)
        User Query
            -> Retrieval           (semantic search, top-k = 5)                  <- here
            -> Generation          (Groq llama-3.3-70b-versatile)   (Milestone 5)
            -> Grounded Response   (with citations)                 (Milestone 5)

What this file gives you:
    * build_index()  — read chunks.json, embed every chunk locally with
      all-MiniLM-L6-v2, and load the vectors + metadata into a persistent
      ChromaDB collection. Run this once (or whenever chunks.json changes).
    * retrieve(query, top_k) — embed a user question with the same model and
      return the top-k most similar review chunks, each with its source
      metadata and cosine distance. This is the function app.py (Milestone 5)
      will call before handing context to the LLM.

Why all-MiniLM-L6-v2 (per planning.md): it runs locally with no API key and no
rate limits, produces 384-dimensional vectors, and is fast and accurate on
short English reviews.

Usage:
    # Build / rebuild the vector store from chunks.json
    python embed.py --build

    # Test retrieval without touching the LLM (this is where retrieval bugs
    # surface — debug them here, before generation, as milestone4.md advises)
    python embed.py --query "Does Polakis assign a lot of homework?"
    python embed.py --query "Who is the most caring professor?" --top-k 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# --- Configuration ----------------------------------------------------------

# Chunk records produced by chunking.py (Milestone 3). This is our input.
CHUNKS_FILE = Path(__file__).parent / "chunks.json"

# Where ChromaDB persists the vector store on disk, so we embed once and reuse
# the index across runs (and from app.py in Milestone 5).
CHROMA_DIR = Path(__file__).parent / "chroma_db"

# The local embedding model named in planning.md. 384-dim vectors, no API key.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Name of the collection inside ChromaDB that holds the review vectors.
COLLECTION_NAME = "uic_professor_reviews"

# Default number of chunks retrieved per query (planning.md: top-k starts at 5).
DEFAULT_TOP_K = 5

# Cache the loaded model so repeated retrieve() calls don't reload it.
_model: SentenceTransformer | None = None


# --- Model + store helpers ---------------------------------------------------

def get_model() -> SentenceTransformer:
    """Load (once) and return the all-MiniLM-L6-v2 embedding model.

    The first call downloads the model weights to the local HuggingFace cache;
    every later call in the same process reuses the in-memory instance.
    """
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBED_MODEL_NAME} ...")
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def get_client() -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client backed by CHROMA_DIR."""
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(create: bool = False) -> chromadb.Collection:
    """Return the reviews collection.

    We pin the similarity metric to cosine ("hnsw:space": "cosine") because the
    MiniLM sentence embeddings are meant to be compared by cosine similarity,
    and cosine distance is the value planning.md says we'll eyeball to judge
    retrieval quality (0 = identical meaning, ~1 = unrelated).
    """
    client = get_client()
    if create:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(name=COLLECTION_NAME)


# --- Embedding + indexing (Phase 1) -----------------------------------------

def load_chunks() -> list[dict]:
    """Load the structured chunk records written by chunking.py."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"{CHUNKS_FILE} not found. Run `python chunking.py` first to "
            "generate the chunks from the documents/ folder."
        )
    return json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))


def flatten_metadata(metadata: dict) -> dict:
    """Make a chunk's metadata safe to store in ChromaDB.

    Chroma metadata values must be scalars (str / int / float / bool), but our
    chunks carry `class_codes` as a list. We join it into a comma-separated
    string so the source attribution survives into the vector store. The two
    fields the assignment requires — the source document name and the chunk's
    position in that document — are kept as-is (`source` and `review_index`).
    """
    return {
        "source": metadata["source"],            # source document name (required)
        "professor": metadata["professor"],
        "review_index": metadata["review_index"],  # position within the document (required)
        "chunk_index": metadata["chunk_index"],    # global position across corpus
        "class_codes": ", ".join(metadata.get("class_codes", [])),
    }


def build_index() -> chromadb.Collection:
    """Embed every chunk and (re)load it into the ChromaDB collection.

    We rebuild from scratch each time so the store always mirrors chunks.json:
    drop any existing collection, embed all chunk texts in one batch with
    all-MiniLM-L6-v2, then add the vectors together with their ids, documents,
    and flattened source metadata.
    """
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE.name}")

    # Start clean so re-running never duplicates or leaves stale vectors.
    client = get_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Removed existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass  # collection didn't exist yet — fine on a first run.

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [flatten_metadata(chunk["metadata"]) for chunk in chunks]

    model = get_model()
    print("Embedding chunks with all-MiniLM-L6-v2 ...")
    embeddings = model.encode(
        documents,
        show_progress_bar=True,
        normalize_embeddings=True,  # unit vectors -> clean cosine distances
    )

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )

    print(
        f"\nIndexed {collection.count()} chunks into ChromaDB "
        f"collection '{COLLECTION_NAME}' at {CHROMA_DIR}"
    )
    return collection


# --- Retrieval (Phase 2) -----------------------------------------------------

def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Return the top-k review chunks most semantically similar to `query`.

    The query is embedded with the SAME model used at index time (this match is
    essential — mixing models would make the distances meaningless). Each result
    carries the chunk text, its source metadata, and the cosine distance so the
    caller (and our manual evaluation) can judge how relevant the hit is.

    This is the function Milestone 5's generation step calls to build grounded,
    cited context for the LLM.
    """
    collection = get_collection()
    model = get_model()

    query_embedding = model.encode([query], normalize_embeddings=True)
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
    )

    # Chroma returns parallel lists wrapped in an outer list (one per query).
    hits: list[dict] = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "text": doc,
                "metadata": meta,
                "distance": distance,        # cosine distance: lower = closer
                "similarity": 1 - distance,  # convenience: higher = closer
            }
        )
    return hits


# --- CLI / inspection --------------------------------------------------------

def print_hit(rank: int, hit: dict) -> None:
    """Pretty-print one retrieval result for manual inspection."""
    meta = hit["metadata"]
    codes = meta.get("class_codes") or "n/a"
    print("-" * 70)
    print(
        f"#{rank}  cosine distance: {hit['distance']:.4f}  "
        f"(similarity {hit['similarity']:.4f})"
    )
    print(
        f"     {meta['professor']}  |  source: {meta['source']}  |  "
        f"review #{meta['review_index']}  |  class(es): {codes}"
    )
    print("-" * 70)
    print(hit["text"])
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed UIC review chunks into ChromaDB and test retrieval."
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Embed chunks.json and (re)build the ChromaDB vector store.",
    )
    parser.add_argument(
        "--query",
        metavar="TEXT",
        help='Run a semantic search, e.g. --query "Does Polakis give a lot of homework?"',
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"How many chunks to retrieve (default: {DEFAULT_TOP_K}).",
    )
    args = parser.parse_args()

    if args.build:
        build_index()

    if args.query:
        # Build the index automatically if the user queries before building.
        try:
            get_collection()
        except Exception:
            print("No vector store found yet — building it first.\n")
            build_index()

        print(f'\nTop {args.top_k} chunks for: "{args.query}"\n')
        for rank, hit in enumerate(retrieve(args.query, args.top_k), start=1):
            print_hit(rank, hit)

    if not args.build and not args.query:
        parser.print_help()


if __name__ == "__main__":
    main()
