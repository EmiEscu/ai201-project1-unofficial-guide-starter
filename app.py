"""
app.py — Phase 2 (tail): Generation + Query Interface

The Unofficial Guide: UIC Course & Professor Reviews (RAG)

This script implements the final two stages of the architecture diagram in
planning.md, and ties the whole pipeline together behind a usable UI:

    Phase 2: Query Pipeline (RAG)
        User Query
            -> Retrieval           (semantic search, top-k = 5)   (embed.py)
            -> Generation          (Groq llama-3.3-70b-versatile)            <- here
            -> Grounded Response   (with citations)                          <- here

The whole point of this milestone is GROUNDING: the LLM must answer using ONLY
the review chunks we retrieved, never its own training knowledge. We enforce
that two ways — and it matters that both are present:

    1. A strict system prompt that *instructs* the model to answer only from the
       provided context and to refuse ("I don't have enough information on
       that.") when the context doesn't cover the question.

    2. Structural guarantees the model cannot undermine:
         * We only ever put retrieved chunks in the prompt — there is no other
           knowledge in the context window.
         * A relevance gate: if even the best-matching chunk is too far away
           (cosine distance too high), we don't call the LLM at all — we return
           the refusal directly. This stops confident answers to off-topic
           questions ("what's the weather?") before generation can happen.
         * Source attribution is appended PROGRAMMATICALLY from the retrieved
           chunks' metadata after generation. We do not trust the model to cite
           correctly — the "Sources" list is built from `retrieve()`'s output,
           so every answer is attributable to real documents by construction.

Output format (per milestone5.md): every grounded response is
    <answer>  +  a "Sources" list naming the document(s) it drew from.

Usage:
    # Launch the Gradio web interface (default)
    python app.py

    # Ask one question from the command line (handy for the evaluation report)
    python app.py --ask "Does Polakis assign a lot of homework?"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from embed import DEFAULT_TOP_K, retrieve

# --- Configuration ----------------------------------------------------------

# Load GROQ_API_KEY from the .env file next to this script.
load_dotenv(Path(__file__).parent / ".env")

# The LLM named in planning.md / the assignment: Groq's free-tier, OpenAI-
# compatible Llama 3.3 70B.
LLM_MODEL = "llama-3.3-70b-versatile"

# How many review chunks to retrieve and feed as context (planning.md: 5).
TOP_K = DEFAULT_TOP_K

# Keep generation tight and factual — we want the documents to speak, not the
# model's prose. Low temperature reduces the chance of the model "filling gaps"
# from its own knowledge.
TEMPERATURE = 0.2
MAX_TOKENS = 700

# Relevance gate. Cosine distance runs 0 (identical meaning) .. ~2 (opposite);
# for these short English reviews, on-topic hits land well under ~0.6 and truly
# unrelated queries sit far higher. If the BEST hit is past this threshold, the
# corpus almost certainly doesn't cover the question, so we refuse instead of
# letting the LLM improvise. Set deliberately lenient so real questions still
# pass; tighten it if off-topic questions slip through.
MAX_RELEVANT_DISTANCE = 0.95

# The exact refusal string. We instruct the model to use it verbatim AND emit it
# ourselves when the relevance gate trips, so the "no answer" path is identical
# whether retrieval or generation decided there wasn't enough to go on.
INSUFFICIENT_INFO = "I don't have enough information on that."

# Grounding instruction. This is the contract with the model: answer ONLY from
# the numbered context, cite the [n] sources used, and refuse rather than guess.
SYSTEM_PROMPT = f"""You are The Unofficial Guide, a question-answering assistant for \
students researching UIC computer-science professors. You answer using ONLY the \
student reviews provided to you in the CONTEXT section of each message.

Strict rules:
1. Use ONLY the information in the provided CONTEXT. Do NOT use any outside or \
prior knowledge about these professors, courses, or UIC. If you happen to "know" \
something that is not in the CONTEXT, you must not use it.
2. If the CONTEXT does not contain enough information to answer the question, \
reply with exactly this sentence and nothing else: "{INSUFFICIENT_INFO}"
3. Do not invent professors, classes, grades, or quotes. Every claim you make \
must be supported by the CONTEXT.
4. The reviews are student opinions, not facts — phrase answers accordingly \
(e.g. "students say...", "several reviews mention...").
5. Cite your sources inline using the bracketed numbers from the CONTEXT, e.g. \
"students found the workload heavy [1][3]".
6. Be concise: 1-4 sentences is usually enough. Do not pad the answer.

Answer the user's question based strictly on the CONTEXT they provide."""


# --- Groq client -------------------------------------------------------------

_client: Groq | None = None


def get_client() -> Groq:
    """Load (once) and return the Groq client, reading GROQ_API_KEY from .env."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key or api_key == "your_key_here":
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and paste "
                "your free key from https://console.groq.com."
            )
        _client = Groq(api_key=api_key)
    return _client


# --- Prompt assembly + source attribution ------------------------------------

def build_context(hits: list[dict]) -> str:
    """Format retrieved chunks into a numbered CONTEXT block for the LLM.

    Each chunk is labelled [n] with its professor and source file so the model
    can cite it inline, and so the numbers line up with the programmatic Sources
    list we build separately. Only this text reaches the model — there is no
    other knowledge in the prompt, which is the structural half of grounding.
    """
    blocks = []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        header = f"[{i}] Professor: {meta['professor']}  (source: {meta['source']})"
        blocks.append(f"{header}\n{hit['text']}")
    return "\n\n".join(blocks)


def snippet(text: str, max_chars: int = 180) -> str:
    """Collapse a review chunk to a one-line preview of its actual comment.

    Review chunks lead with a RateMyProfessor metadata header
    ("Quality / <score> / Difficulty / ... / Textbook: <value>") followed by the
    free-text comment and tags. The header is the same boilerplate on every
    review, so we skip past it to the comment — that's what helps a reader
    recognize which review a citation points to. We flatten whitespace and
    truncate; this is only a hint back to the real review, not its full text. If
    the expected header isn't found, we fall back to the whole flattened chunk.
    """
    flat = " ".join(text.split())

    # The metadata block always ends with "Textbook: <value>"; the comment
    # follows it. Split on the LAST occurrence in case a comment mentions it.
    marker = "Textbook:"
    if marker in flat:
        after = flat.rsplit(marker, 1)[1].strip()
        # Drop the textbook value token itself (e.g. "No", "Yes", "N/A").
        parts = after.split(" ", 1)
        if len(parts) == 2 and parts[1].strip():
            flat = parts[1].strip()

    return flat if len(flat) <= max_chars else flat[:max_chars].rstrip() + "…"


def format_sources(hits: list[dict]) -> str:
    """Build the Sources list PROGRAMMATICALLY from the retrieved chunks.

    Source attribution is guaranteed here, not left to the model: every entry is
    read straight from `retrieve()`'s output. We list ONE entry per retrieved
    chunk, numbered [1]..[k] in the SAME order build_context() fed them to the
    LLM — so the inline [n] citations in the answer point to a visible, numbered
    review the reader can actually inspect. Each line shows the professor, source
    file, similarity (1 - cosine distance, higher = closer), and a short snippet
    of the review the claim was drawn from.
    """
    lines = ["**Sources**", ""]
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        lines.append(
            f"**[{i}]** {meta['professor']} — `{meta['source']}` "
            f"(relevance {hit['similarity']:.2f})  "
            f"\n> {snippet(hit['text'])}"
        )
    return "\n".join(lines)


# --- Generation --------------------------------------------------------------

def generate_answer(query: str, hits: list[dict]) -> str:
    """Call Groq's Llama 3.3 70B with the grounded prompt and return its text."""
    context = build_context(hits)
    user_message = (
        f"CONTEXT (student reviews retrieved for this question):\n\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer using ONLY the CONTEXT above. If it is insufficient, reply "
        f'exactly: "{INSUFFICIENT_INFO}"'
    )

    response = get_client().chat.completions.create(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content.strip()


def answer_query(query: str, top_k: int = TOP_K) -> str:
    """Full RAG step: retrieve -> (gate) -> generate -> attach sources.

    Returns a single Markdown string in the required output format:
    a grounded answer followed by a programmatic Sources list. When the corpus
    can't support the question (empty query, no relevant chunks, or the model
    refuses), it returns the refusal WITHOUT a Sources list — there is nothing
    real to attribute the (non-)answer to.
    """
    query = (query or "").strip()
    if not query:
        return "Please enter a question about a UIC CS professor."

    hits = retrieve(query, top_k=top_k)

    # Structural grounding: if nothing came back, or even the closest chunk is
    # too far in meaning, refuse before involving the LLM.
    if not hits or hits[0]["distance"] > MAX_RELEVANT_DISTANCE:
        return INSUFFICIENT_INFO

    answer = generate_answer(query, hits)

    # If the model refused, don't tack on sources for an answer it didn't give.
    if INSUFFICIENT_INFO.rstrip(".").lower() in answer.rstrip(".").lower():
        return INSUFFICIENT_INFO

    return f"{answer}\n\n{format_sources(hits)}"


# --- Gradio interface --------------------------------------------------------

def build_interface():
    """Build the Gradio app: a question box, an answer pane, and examples."""
    import gradio as gr

    example_questions = [
        "Does Polakis assign a lot of homework?",
        "Is Anastasios Sidiropoulos approachable if I have questions?",
        "What do students praise about Luis Pina's class structure?",
        "Which professor has the highest overall rating?",
        "Are there exams in Natalie Parde's class?",
    ]

    with gr.Blocks(title="The Unofficial Guide — UIC CS Professors") as demo:
        gr.Markdown(
            "# 🎓 The Unofficial Guide — UIC CS Professors\n"
            "Ask a plain-language question about a UIC computer-science professor. "
            "Answers are grounded **only** in real RateMyProfessor reviews and come "
            "with the sources they were drawn from. If the reviews don't cover your "
            "question, the guide will say so rather than guess."
        )

        with gr.Row():
            question = gr.Textbox(
                label="Your question",
                placeholder="e.g. Does Polakis assign a lot of homework?",
                lines=2,
                scale=4,
            )
            ask_btn = gr.Button("Ask", variant="primary", scale=1)

        answer = gr.Markdown(label="Answer")

        gr.Examples(examples=example_questions, inputs=question)

        # Wire both the button and the Enter key to the same handler.
        ask_btn.click(fn=answer_query, inputs=question, outputs=answer)
        question.submit(fn=answer_query, inputs=question, outputs=answer)

    return demo


# --- Entry point -------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grounded Q&A over UIC professor reviews (RAG + Groq)."
    )
    parser.add_argument(
        "--ask",
        metavar="QUESTION",
        help="Answer one question on the command line instead of launching the UI.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"How many chunks to retrieve as context (default: {TOP_K}).",
    )
    args = parser.parse_args()

    if args.ask:
        print(answer_query(args.ask, top_k=args.top_k))
        return

    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
