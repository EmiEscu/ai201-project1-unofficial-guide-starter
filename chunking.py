"""
chunking.py — Phase 1 (ETL): Document Ingestion + Chunking

The Unofficial Guide: UIC Course & Professor Reviews (RAG)

This script implements the Chunking Strategy described in planning.md:

    Each RateMyProfessor review in a professor's .txt file is one self-contained
    review. Reviews are separated by a blank line (\\n\\n). Every review follows
    the same shape:

        Quality                     <- always the FIRST line of a review
        <numeric quality score>
        Difficulty
        <numeric difficulty score>
        <class code / date / metadata lines>
        <free-text comment>
        <descriptor tags>           <- always the LAST line (e.g.
                                       "Caring", "Tough grader", "Group projects")

    So we chunk by splitting on blank lines. There is NO overlap, because each
    blank line cleanly separates one complete review from the next. The number
    of lines / characters / tokens inside a review does not matter — what matters
    is that a chunk starts at "Quality" and ends at the descriptor tags.

Pipeline position:
    Document Ingestion (raw .txt) -> [chunking.py] -> Embedding -> Vector Store

Output:
    chunks.json — a list of chunk records, each with the review text plus
    metadata (source file, professor name, class codes, chunk index). This file
    is the input to the embedding + ChromaDB stage (Milestone 4).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# --- Configuration ----------------------------------------------------------

# Folder holding the raw professor review documents.
DOCUMENTS_DIR = Path(__file__).parent / "documents"

# Where the structured chunks get written for the embedding stage.
OUTPUT_FILE = Path(__file__).parent / "chunks.json"

# A blank line (optionally with stray spaces/tabs) marks the boundary between
# two reviews. Matches one OR MORE blank lines so accidental double-spacing in
# the source files still produces clean splits.
REVIEW_SEPARATOR = re.compile(r"\n[ \t]*\n+")

# Matches a UIC class code like "CS107", "CS468", "CS494". Used only to enrich
# the chunk metadata (helps later retrieval/filtering); it does not affect how
# we split.
CLASS_CODE = re.compile(r"\bCS\d{3}\b")


# --- Core chunking logic -----------------------------------------------------

def clean_review(chunk: str) -> str:
    """Normalize a single review block.

    Some source files contain stray scraped lines before the review (e.g. a
    leading 'Reviewed: Dec 10th, 2024' line). A real review always begins at
    the 'Quality' header, so if there is text before the first 'Quality', we
    drop it. This keeps every chunk in the expected
    'starts with Quality ... ends with descriptor tags' shape.
    """
    chunk = chunk.strip()
    idx = chunk.find("Quality")
    if idx > 0:
        chunk = chunk[idx:].strip()
    return chunk


def chunk_document(text: str) -> list[str]:
    """Split one professor's document into individual review chunks.

    A chunk is the text between blank lines. We strip surrounding whitespace,
    clean any stray pre-review lines, and drop empty fragments so trailing
    newlines / leading blank lines in the file don't create empty chunks.
    """
    raw_chunks = REVIEW_SEPARATOR.split(text.strip())
    cleaned = [clean_review(chunk) for chunk in raw_chunks]
    return [chunk for chunk in cleaned if chunk]


def is_valid_review(chunk: str) -> bool:
    """A well-formed review chunk starts with the 'Quality' header.

    Per the chunking strategy, the way to know a chunk captured a whole review
    is that it begins with 'Quality'. We use this as a sanity check and warn on
    anything that doesn't match (e.g. a stray header or malformed block).
    """
    return chunk.lstrip().startswith("Quality")


def professor_name_from_filename(path: Path) -> str:
    """Turn 'Jason_Polakis.txt' into 'Jason Polakis' for clean attribution."""
    return path.stem.replace("_", " ").strip()


def extract_class_codes(chunk: str) -> list[str]:
    """Pull any UIC class codes (CS###) mentioned in the review, de-duplicated."""
    return sorted(set(CLASS_CODE.findall(chunk)))


def build_chunk_records(documents_dir: Path) -> list[dict]:
    """Load every .txt document and produce structured chunk records."""
    txt_files = sorted(documents_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt documents found in {documents_dir}. "
            "Make sure your professor review files are in the documents/ folder."
        )

    records: list[dict] = []
    global_index = 0

    for path in txt_files:
        professor = professor_name_from_filename(path)
        text = path.read_text(encoding="utf-8")
        reviews = chunk_document(text)

        for local_index, review in enumerate(reviews):
            if not is_valid_review(review):
                print(
                    f"  [warn] {path.name} chunk #{local_index} does not start "
                    f"with 'Quality' — check the source spacing:\n"
                    f"        {review[:60]!r}..."
                )

            records.append(
                {
                    # Stable unique id for the vector store.
                    "id": f"{path.stem}_{local_index}",
                    "text": review,
                    "metadata": {
                        "source": path.name,
                        "professor": professor,
                        "class_codes": extract_class_codes(review),
                        "review_index": local_index,
                        "chunk_index": global_index,
                    },
                }
            )
            global_index += 1

        print(f"  {path.name:<35} -> {len(reviews):>3} review chunks ({professor})")

    return records


# --- Inspection helpers ------------------------------------------------------

def print_chunk(record: dict) -> None:
    """Pretty-print one chunk record with its professor name and metadata."""
    meta = record["metadata"]
    print("-" * 60)
    print(f"Professor : {meta['professor']}")
    print(f"Source    : {meta['source']}")
    print(f"Review #   : {meta['review_index']}  (global chunk #{meta['chunk_index']})")
    print(f"Class(es) : {', '.join(meta['class_codes']) or 'n/a'}")
    print("-" * 60)
    print(record["text"])
    print("-" * 60)


def show_chunk(records: list[dict], professor: str, review_index: int) -> None:
    """Find and print review `review_index` for the given professor.

    Matching on the professor name is case-insensitive and matches on a partial
    name, so `--show pina 10` works as well as `--show "Luis Pina" 10`.
    """
    needle = professor.lower()
    matches = [r for r in records if needle in r["metadata"]["professor"].lower()]

    if not matches:
        names = sorted({r["metadata"]["professor"] for r in records})
        print(f"No professor matching {professor!r}. Available professors:")
        for name in names:
            print(f"  - {name}")
        return

    target = [r for r in matches if r["metadata"]["review_index"] == review_index]
    if not target:
        prof_name = matches[0]["metadata"]["professor"]
        print(
            f"{prof_name} has {len(matches)} reviews (indexes 0–{len(matches) - 1}); "
            f"there is no review #{review_index}."
        )
        return

    print_chunk(target[0])


def show_all_chunks(records: list[dict], professor: str) -> None:
    """Print every chunk for the given professor, in review order.

    Like `show_chunk`, the professor match is case-insensitive and partial, so
    `--all pina` works as well as `--all "Luis Pina"`.
    """
    needle = professor.lower()
    matches = [r for r in records if needle in r["metadata"]["professor"].lower()]

    if not matches:
        names = sorted({r["metadata"]["professor"] for r in records})
        print(f"No professor matching {professor!r}. Available professors:")
        for name in names:
            print(f"  - {name}")
        return

    matches.sort(key=lambda r: r["metadata"]["review_index"])
    prof_name = matches[0]["metadata"]["professor"]
    print(f"All {len(matches)} chunks for {prof_name}:\n")
    for record in matches:
        print_chunk(record)
        print()


# --- Entry point -------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk UIC professor reviews by blank-line spacing."
    )
    parser.add_argument(
        "--show",
        nargs=2,
        metavar=("PROFESSOR", "REVIEW_INDEX"),
        help="Inspect one chunk, e.g. --show \"Luis Pina\" 10  (reviews are 0-indexed)",
    )
    parser.add_argument(
        "--all",
        metavar="PROFESSOR",
        help="Print every chunk for one professor in order, e.g. --all \"Luis Pina\"",
    )
    args = parser.parse_args()

    print(f"Loading documents from: {DOCUMENTS_DIR}\n")
    records = build_chunk_records(DOCUMENTS_DIR)

    OUTPUT_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    professors = {r["metadata"]["professor"] for r in records}
    valid = sum(1 for r in records if is_valid_review(r["text"]))

    print("\n" + "-" * 60)
    print(f"Professors processed : {len(professors)}")
    print(f"Total review chunks  : {len(records)}")
    print(f"Well-formed chunks   : {valid}/{len(records)} start with 'Quality'")
    print(f"Wrote chunks to      : {OUTPUT_FILE}")
    print("-" * 60)

    # If the user asked for a specific chunk, show it. Otherwise show the first
    # chunk so you can eyeball that a whole review was captured (starts at
    # 'Quality', ends at descriptor tags) — the manual verification the planning
    # doc calls for.
    if args.all:
        print()
        show_all_chunks(records, args.all)
    elif args.show:
        professor, raw_index = args.show
        try:
            review_index = int(raw_index)
        except ValueError:
            print(f"\nREVIEW_INDEX must be a number, got {raw_index!r}.")
            return
        print(f"\nRequested chunk — {professor}, review #{review_index}:\n")
        show_chunk(records, professor, review_index)
    elif records:
        print("\nSample chunk (first review):\n")
        print_chunk(records[0])


if __name__ == "__main__":
    main()
